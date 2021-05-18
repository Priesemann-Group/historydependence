import argparse
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
import h5py
import csv
import ast
import yaml
import numpy as np
from scipy.optimize import minimize, bisect
from scipy.io import loadmat
import pandas as pd

CODE_DIR = dirname(realpath(__file__))
path.insert(1, '{}/src'.format(CODE_DIR))

if 'hde_fast_glm' not in modules:
    from hde_fast_glm import*
    import hde_utils as utl

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

def get_temporal_depth(rec_length, run_index, ANALYSIS_DIR, regularization_method = 'bbc'):
    merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    merged_csv_file = open(merged_csv_file_name, 'r')
    # Find the temporal depth for given rec_length and sample_index
    line_index = 0
    for label in utl.load_from_CSV_file(merged_csv_file, 'label'):
        rec_length_label = label.split("-")[0]
        run_index_label = int(label.split("-")[2])
        if rec_length_label == rec_length and run_index == run_index_label:
            temporal_depth = float(utl.load_from_CSV_file(
                merged_csv_file, 'T_D_{}'.format(regularization_method))[line_index])
            break
        line_index += 1
    merged_csv_file.close()
    return temporal_depth

def load_embedding_parameters(rec_length, sample_index, ANALYSIS_DIR, regularization_method = 'bbc'):
    prefix = 'ANALYSIS'
    merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(merged_csv_file_name)
    # Find the analysis num for given rec_length and sample_index
    line_index = 0
    for label in statistics_pd['label']:
        rec_length_label = label.split("-")[0]
        sample_index_label = int(label.split("-")[2])
        if rec_length_label == rec_length and sample_index == sample_index_label:
            analysis_num = int(statistics_pd['#analysis_num'][line_index])
            analysis_num_str = str(analysis_num)
            for i in range(4 - len(str(analysis_num))):
                analysis_num_str = '0' + analysis_num_str
            print(sample_index, analysis_num_str)
            break
        line_index += 1
    # Load the histdep_csv to extract the embeddings for every past range
    histdep_csv_file_name = '{}/{}/histdep_data.csv'.format(
        ANALYSIS_DIR, prefix + analysis_num_str)
    histdep_csv_file = open(histdep_csv_file_name, 'r')
    T = utl.load_from_CSV_file(histdep_csv_file, 'T')
    d = utl.load_from_CSV_file(histdep_csv_file, 'number_of_bins_d_{}'.format(regularization_method))
    kappa = utl.load_from_CSV_file(histdep_csv_file, 'scaling_k_{}'.format(regularization_method))
    tau = utl.load_from_CSV_file(histdep_csv_file, 'first_bin_size_{}'.format(regularization_method))
    embedding_parameters = np.array([T, d, kappa, tau])
    histdep_csv_file.close()
    return embedding_parameters, analysis_num_str

def get_embeddings_for_optimization(past_range, d, max_first_bin_size):
    uniform_bin_size = past_range / d
    if d == 1:
        tau = uniform_bin_size
        kappa = 0.0
    else:
        if uniform_bin_size <= max_first_bin_size:
            # if uniform bins are small enough, than use uniform embedding
            tau = uniform_bin_size
            kappa = 0.0
        else:
             # If the bin size with uniform bins is > max_first_bin_size, then choose exponential embedding such that first bin size is equal to minimum first bin size
            tau = max_first_bin_size
            kappa = bisect(lambda k: np.sum(
                max_first_bin_size * np.power(10, np.arange(d) * k)) - past_range, -1., 10)
    return kappa, tau

def preprocess_spiketimes(spiketimes, glm_settings):
    T_0 = float(glm_settings['burning_in_time'])
    t_bin = float(glm_settings['embedding_step_size'])
    T_f = T_0 + float(glm_settings['total_recording_length'])
    # Shift spiketimes to ignore spikes from the burn in period, because analysis starts at time 0
    spiketimes = spiketimes - T_0
    # computes binary spike counts to represent current spiking
    counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
    return spiketimes, counts

def load_and_preprocess_spiketimes_experiments(recorded_system, neuron_index, glm_settings, data_path):
    DATA_DIR = '{}/{}'.format(data_path, recorded_system)
    validNeurons = np.load(
        '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)
    neuron = validNeurons[neuron_index]
    spiketimes = np.load(
        '{}/spks/spiketimes_neuron{}.npy'.format(DATA_DIR, neuron))
    t_bin = float(glm_settings['embedding_step_size'])
    # Offset starting time of the analysis such that at least 5 seconds of spiking history are observed
    T_0 = spiketimes[1] + 5.
    T_f_recording = spiketimes[-1] - 2.
    T_f_max = T_0 + float(glm_settings['total_recording_length'])
    T_f = np.amin([T_f_recording, T_f_max])
    # Shift spiketimes by starting time T_0, because analysis starts at time 0
    spiketimes = spiketimes - T_0
    # computes binary spike counts to represent current spiking
    counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
    return spiketimes, counts

def save_glm_benchmark_to_CSV(glm_benchmark, embedding_parameters, ANALYSIS_DIR, analysis_num_str, regularization_method='bbc'):
    ANALYSIS_DIR = ANALYSIS_DIR + '/ANALYSIS' + analysis_num_str
    glm_csv_file_name = '{}/glm_benchmark_{}.csv'.format(
        ANALYSIS_DIR, regularization_method)
    with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM"])
        writer.writeheader()
        for i, T in enumerate(embedding_parameters[0]):
            writer.writerow(
                {"T": T, "number_of_bins_d": int(embedding_parameters[1][i]), "scaling_kappa": embedding_parameters[2][i], "first_bin_size": embedding_parameters[3][i], "R_GLM": glm_benchmark[i]})
    return EXIT_SUCCESS

def save_glm_estimates_to_CSV_Simulation(past_range, glm_estimates, BIC, glm_settings, ANALYSIS_DIR):
    glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR)
    with open(glm_csv_file_name, 'a+', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size","embedding_mode_optimization", "BIC", "R_GLM"])
        for i, d in enumerate(glm_settings['embedding_number_of_bins_set']):
            d = int(d)
            max_first_bin_size = float(glm_settings['max_first_bin_size'])
            embedding_mode_optimization = glm_settings['embedding_mode_optimization']
            kappa, tau = get_embeddings_for_optimization(
                past_range, d, max_first_bin_size)
            writer.writerow(
                {"T": past_range, "number_of_bins_d": d, "scaling_kappa": kappa, "first_bin_size": tau, "embedding_mode_optimization": embedding_mode_optimization, "BIC": BIC[i], "R_GLM": glm_estimates[i]})
    return EXIT_SUCCESS

def save_glm_estimates_to_CSV_Experiments(past_range, opt_embedding_parameters, glm_estimates, BIC, glm_settings, ANALYSIS_DIR, analysis_num_str):
    ANALYSIS_DIR = ANALYSIS_DIR + '/ANALYSIS' + analysis_num_str
    glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR)
    with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "BIC", "R_GLM"])
        writer.writeheader()
        d = int(opt_embedding_parameters[1][0])
        kappa = opt_embedding_parameters[2][0]
        tau = opt_embedding_parameters[3][0]
        writer.writerow(
            {"T": past_range, "number_of_bins_d": d, "scaling_kappa": kappa, "first_bin_size": tau, "BIC": BIC[0], "R_GLM": glm_estimates[0]})
        for i, d in enumerate(glm_settings['embedding_number_of_bins_set']):
            d = int(d)
            max_first_bin_size = float(glm_settings['max_first_bin_size'])
            kappa, tau = get_embeddings_for_optimization(
                past_range, d, max_first_bin_size)
            writer.writerow(
                {"T": past_range, "number_of_bins_d": d, "scaling_kappa": kappa, "first_bin_size": tau, "BIC": BIC[i+1], "R_GLM": glm_estimates[i+1]})
    return EXIT_SUCCESS


def fit_GLM_params(counts, past, d, N_bins):
    mu_0 = 1.
    h_0 = np.zeros(d)
    res = minimize(lambda param: -L_B_past(counts, past, d, N_bins, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
        counts, past, d, N_bins, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts, past, d, N_bins, param[1:], param[0]))
    mu = res.x[0]
    h = res.x[1:]
    return mu, h


def compute_R_GLM(counts, past, d, N, mu, h):
    P_spike = np.sum(counts) / float(N)
    H_spike = -P_spike * np.log(P_spike) - (1 - P_spike) * np.log(1 - P_spike)
    R_GLM = 1 - H_cond_B_past(counts, past, d, N, mu, h) / H_spike
    return R_GLM


def compute_BIC_GLM(counts, past, d, N, mu, h):
    BIC = -2 * L_B_past(counts, past, d, N, h, mu) + (d + 1) * np.log(N)
    return BIC


def compute_benchmark(embedding_parameters, spiketimes, counts, glm_settings):
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_benchmark']
    # Number of total data points
    N = len(counts)
    # Number of training data points and indices for training data set
    N_training = int(N / 3)  # train on one third of the data
    np.random.seed(42)
    training_indices = np.random.choice(N, N_training, replace=False)
    counts_training = counts[training_indices]

    R_GLM = []
    for i, T in enumerate(embedding_parameters[0]):
        # embedding parameters
        d = int(embedding_parameters[1][i])
        kappa = embedding_parameters[2][i]
        tau = embedding_parameters[3][i]

        # apply past embedding
        past = past_activity(spiketimes, d, kappa, tau,
                             t_bin, N, embedding_mode)
        # downsample to obtain smaller training data set to speed up fitting
        past_training = downsample_past_activity(
            past, training_indices, N, d)
        # fit GLM parameters
        mu, h = fit_GLM_params(
            counts_training, past_training, d, N_training)
        # estimate history dependence for fitted GLM parameters
        result = compute_R_GLM(counts, past, d, N, mu, h)
        R_GLM += [result]
        print(T, d, kappa, result)
    return R_GLM

# Compute GLM estimates of R and BIC for set for given past ranges T and a set of embedding dimensions d
def compute_estimates_Simulation(past_range, spiketimes, counts, glm_settings):
    # Load embedding parameters for optimization
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_optimization']
    embedding_number_of_bins_set = np.array(
        glm_settings['embedding_number_of_bins_set']).astype(int)
    max_first_bin_size = float(glm_settings['max_first_bin_size'])
    # Number of total data points
    N = len(counts)
    # Number of training data points and indices for training data set
    N_training = int(N / 3)  # train on one third of the data
    np.random.seed(42)
    training_indices = np.random.choice(N, N_training, replace=False)
    counts_training = counts[training_indices]
    R_GLM = []
    BIC = []
    for d in embedding_number_of_bins_set:
        # get remaining embedding parameters such that the embedding has a certain minimum resolution (set by max_first_bin_size)
        kappa, tau = get_embeddings_for_optimization(
            past_range, d, max_first_bin_size)
        # apply past embedding
        past = past_activity(spiketimes, d, kappa, tau,
                             t_bin, N, embedding_mode)
        past_training = downsample_past_activity(
            past, training_indices, N, d)
        # Fit GLM parameters on smaller training set
        mu, h = fit_GLM_params(
            counts_training, past_training, d, N_training)
        # Fit GLM parameters on the whole data set
        # mu, h = fit_GLM_params(counts, past, d, N)
        # # estimate history dependence for fitted GLM parameters
        # Evaluate GLM estimate of R for fitted GLM parameters on the whole data set
        R_GLM += [compute_R_GLM(counts, past, d, N, mu, h)]
        # Compute BIC for fitted GLM parameters on whole data set
        BIC += [compute_BIC_GLM(counts, past, d, N, mu, h)]
    return R_GLM, BIC


# Compute estimates of R and the Bayesian information criterion (BIC) for given past range and a set of embedding dimensions d

def compute_estimates_Experiments(past_range, opt_embedding_parameters, spiketimes, counts, glm_settings):
    # Load embedding parameters for optimization
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_optimization']
    embedding_number_of_bins_set = np.array(
        glm_settings['embedding_number_of_bins_set']).astype(int)
    max_first_bin_size = float(glm_settings['max_first_bin_size'])
    # Number of total data points
    N = len(counts)
    # Fit GLM for optimal embedding parameters
    d = int(opt_embedding_parameters[1][0])
    kappa = opt_embedding_parameters[2][0]
    tau = opt_embedding_parameters[3][0]
    # apply past embedding
    past = past_activity(spiketimes, d, kappa, tau,
                         t_bin, N, embedding_mode)
    # fit GLM parameters
    mu, h = fit_GLM_params(
        counts, past, d, N)
    # estimate history dependence for fitted GLM parameters
    R_GLM = [compute_R_GLM(counts, past, d, N, mu, h)]
    BIC = [compute_BIC_GLM(counts, past, d, N, mu, h)]
    for d in embedding_number_of_bins_set:
        # get remaining embedding parameters such that the embedding has a certain minimum resolution (set by max_first_bin_size)
        kappa, tau = get_embeddings_for_optimization(
            past_range, d, max_first_bin_size)
        # apply past embedding
        past = past_activity(spiketimes, d, kappa, tau,
                             t_bin, N, embedding_mode)
        # Fit GLM parameters on the whole data set
        mu, h = fit_GLM_params(counts, past, d, N)
        # Evaluate GLM estimate of R for fitted GLM parameters on the whole data set
        R_GLM += [compute_R_GLM(counts, past, d, N, mu, h)]
        # Compute BIC for fitted GLM parameters on whole data set
        BIC += [compute_BIC_GLM(counts, past, d, N, mu, h)]
    return R_GLM, BIC
