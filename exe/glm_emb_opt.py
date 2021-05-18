import os
import sys
from sys import exit, argv, path
from os.path import realpath, dirname
import csv
import yaml
import numpy as np

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in sys.modules:
    import hde_glm as glm

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

"""Run parameters"""
device_or_run_index = argv[1]
recorded_system = argv[2]

if len(argv) > 5:
    data_path = argv[5]
else:
    data_path = '{}/data'.format(CODE_DIR)

if device_or_run_index == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

def main_Simulation():
    # Get run index for computation on the cluster
    if device_or_run_index == 'cluster':
        past_range_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        past_range_index = int(device_or_run_index)
    # Load settings
    with open('{}/settings/{}_glm.yaml'.format(CODE_DIR, recorded_system), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    if use_settings_path == True:
        ANALYSIS_DIR = glm_settings['ANALYSIS_DIR']
    else:
        ANALYSIS_DIR = '{}/analysis/{}/glm_ground_truth'.format(CODE_DIR, recorded_system)
    # Create csv with header
    if past_range_index == 0:
        glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR)
        with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
            writer = csv.DictWriter(glm_csv_file, fieldnames=[
                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size",  "embedding_mode_optimization", "BIC", "R_GLM"])
            writer.writeheader()
    # Load the 900 minute simulated recording
    DATA_DIR = '{}/{}'.format(data_path, recorded_system)
    spiketimes = np.load('{}/spiketimes_900min.npy'.format(DATA_DIR))
    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(
        spiketimes, glm_settings)

    # Get the past range for which R should be estimated
    embedding_past_range_set = np.array(
        glm_settings['embedding_past_range_set']).astype(float)
    past_range = embedding_past_range_set[past_range_index]
    # Compute optimized history dependence for given past range

    glm_estimates, BIC = glm.compute_estimates_Simulation(past_range, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_estimates_to_CSV_Simulation(
        past_range, glm_estimates, BIC, glm_settings, ANALYSIS_DIR)

    return EXIT_SUCCESS


def main_Experiments():
    rec_length = '90min'
    # Get run index for computation on the cluster
    if device_or_run_index == 'cluster':
        neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        neuron_index = int(device_or_run_index)
    # Load settings
    with open('{}/settings/{}_glm.yaml'.format(CODE_DIR, recorded_system), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    if use_settings_path == True:
        ANALYSIS_DIR = glm_settings['ANALYSIS_DIR']
    else:
        ANALYSIS_DIR = '{}/analysis/{}/analysis_full_bbc'.format(CODE_DIR, recorded_system)
    # Load and preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.load_and_preprocess_spiketimes_experiments(
        recorded_system, neuron_index, glm_settings, data_path)

    # Get the past range for which R should be estimated
    temporal_depth_bbc = glm.get_temporal_depth(
        rec_length, neuron_index, ANALYSIS_DIR, regularization_method = 'bbc')

    embedding_parameters_bbc, analysis_num_str = glm.load_embedding_parameters(
        rec_length, neuron_index, ANALYSIS_DIR, regularization_method = 'bbc')
    embedding_parameters_bbc = embedding_parameters_bbc[:,
                                                        embedding_parameters_bbc[0] == temporal_depth_bbc]

    # Compute optimized estimate of R for given past range
    glm_estimates, BIC = glm.compute_estimates_Experiments(
        temporal_depth_bbc, embedding_parameters_bbc, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_estimates_to_CSV_Experiments(
        temporal_depth_bbc, embedding_parameters_bbc, glm_estimates, BIC, glm_settings, ANALYSIS_DIR, analysis_num_str)

    return EXIT_SUCCESS


if __name__ == "__main__":
    if (recorded_system == 'glif_22s_kernel' or recorded_system == 'glif_1s_kernel'):
        exit(main_Simulation())
    else:
        exit(main_Experiments())
