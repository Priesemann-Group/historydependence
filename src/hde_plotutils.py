import numpy as np
import pandas as pd
import yaml

def load_analysis_results(recorded_system, rec_length, run_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = False):
    if use_settings_path:
        with open('{}/settings/{}_{}.yaml'.format(CODE_DIR, recorded_system, setup), 'r') as analysis_settings_file:
            analysis_settings = yaml.load(
                analysis_settings_file, Loader=yaml.BaseLoader)
        ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    else:
        ANALYSIS_DIR = '{}/analysis/{}/analysis_{}'.format(CODE_DIR, recorded_system, setup)
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)
    index_list = np.where(statistics_pd['label'] == rec_length + "-" + setup + "-" + str(run_index))[0]
    if len(index_list)==0:
        return None
    else:
        index = index_list[0]
        analysis_num_str = str(statistics_pd['#analysis_num'][index])
        for i in range(4 - len(analysis_num_str)):
            analysis_num_str = '0' + analysis_num_str
        # Get T, R(T) plus confidence intervals
        hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
            ANALYSIS_DIR, analysis_num_str)
        hisdep_pd = pd.read_csv(hisdep_csv_file_name)
        R = np.array(hisdep_pd['max_R_{}'.format(regularization_method)])
        R_CI_lo = np.array(hisdep_pd['max_R_{}_CI_lo'.format(regularization_method)])
        R_CI_hi = np.array(hisdep_pd['max_R_{}_CI_hi'.format(regularization_method)])
        R_tot = statistics_pd['R_tot_{}'.format(regularization_method)][index]
        T_D = statistics_pd['T_D_{}'.format(regularization_method)][index]
        if not len(R) == 0:
            T_D = float(T_D)
        T = np.array(hisdep_pd['#T'])
        return ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi

def load_total_mutual_information(recorded_system, rec_length, run_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = False):
    if use_settings_path:
        with open('{}/settings/{}_{}.yaml'.format(CODE_DIR, recorded_system, setup), 'r') as analysis_settings_file:
            analysis_settings = yaml.load(
                analysis_settings_file, Loader=yaml.BaseLoader)
        ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    else:
        ANALYSIS_DIR = '{}/analysis/{}/analysis_{}'.format(CODE_DIR, recorded_system, setup)
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)
    index_list = np.where(statistics_pd['label'] == rec_length + "-" + setup + "-" + str(run_index))[0]
    if len(index_list)==0:
        return None
    else:
        index = index_list[0]
        analysis_num_str = str(statistics_pd['#analysis_num'][index])
        for i in range(4 - len(analysis_num_str)):
            analysis_num_str = '0' + analysis_num_str
        # Get T, R(T) plus confidence intervals
        hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
            ANALYSIS_DIR, analysis_num_str)
        hisdep_pd = pd.read_csv(hisdep_csv_file_name)
        I_tot = statistics_pd['AIS_tot_{}'.format(regularization_method)][index]
        return I_tot


def load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str):
    glm_csv_file_name = '{}/ANALYSIS{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_pd = pd.read_csv(glm_csv_file_name)
    index = np.argmin(glm_pd['BIC'])
    R_tot_glm = glm_pd['R_GLM'][index]
    return R_tot_glm

def load_analysis_results_glm_Simulation(CODE_DIR, recorded_system, use_settings_path = False):
    if use_settings_path:
        with open('{}/settings/{}_glm.yaml'.format(CODE_DIR, recorded_system), 'r') as glm_settings_file:
            glm_settings = yaml.load(
                glm_settings_file, Loader=yaml.BaseLoader)
        ANALYSIS_DIR = glm_settings["ANALYSIS_DIR"]
    else:
        ANALYSIS_DIR = '{}/analysis/{}/glm_ground_truth'.format(CODE_DIR,recorded_system)
    glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR)
    glm_pd = pd.read_csv(glm_csv_file_name)
    R_glm = []
    for T in np.sort(np.unique(glm_pd["T"])):
        indices = np.where(glm_pd["T"]==T)[0]
        # optimal embedding maximizes the estimated history dependence (on the full recording) set for given past range
        opt_index = np.argmax(glm_pd["R_GLM"][indices])
        R_glm += [glm_pd["R_GLM"][indices][opt_index]]
    return np.sort(np.unique(glm_pd["T"])), np.array(R_glm)

def get_CI_median(samples):
    N_samples = len(samples)
    median = np.median(samples)
    median_samples = np.sort(np.median(np.random.choice(
        samples, size=(10000, N_samples)), axis=1))
    CI_lo = median_samples[249]
    CI_hi = median_samples[9749]
    return CI_lo, CI_hi


def get_CI_mean(samples):
    N_samples = len(samples)
    mean = np.mean(samples)
    mean_samples = np.sort(np.mean(np.random.choice(
        samples, size=(10000, N_samples)), axis=1))
    CI_lo = mean_samples[249]
    CI_hi = mean_samples[9749]
    return CI_lo, CI_hi


def get_R_tot(T, R, R_CI_lo):
    R_max = np.amax(R)
    R_CI_lo[np.isnan(R_CI_lo)] = R[np.isnan(R_CI_lo)]
    if len(np.nonzero(R-R_CI_lo)[0]) == 0:
        return None
    else:
        std_R_max = (R_max - R_CI_lo[R == np.amax(R)][np.nonzero(R_max - R_CI_lo[R == np.amax(R)])][0])/2
        T_D = T[R > R_max - std_R_max][0]
        T_max_valid = T[R > R_max - std_R_max][-1]
        T_D_index = np.where(T == T_D)[0][0]
        max_valid_index = np.where(T == T_max_valid)[0][0]+1
        R_tot = np.mean(R[T_D_index:max_valid_index])
        return R_tot, T_D_index, max_valid_index

def get_T_avg_old(T,R,R_tot):
    # R_max = np.amax(R)
    R_prev = 0.
    T_avg_short = 0
    T_avg_long = 0
    R_thresh = (1-1/np.exp(1))*R_tot
    # I added this as an exeption to handle the true model values better (otherwise they are not correctly normalized, because we only compute R(T) up to T = 3sec)
    if R_tot > np.amax(R):
        R_tot = np.amax(R)
    for i, T_i in enumerate(T):
        # Is T_i > T_1?
        if R[i]>R_thresh:
            if R[i]>R_tot:
                R_val = R_tot
            else:
                R_val= R[i]
            dR = np.amax([0.0,R_val-R_prev])
            if R[i-1]<R_thresh:
                R_half =  R[i]
                R_diff_half = (T_i + T[i-1])/2 * dR/(T_i - T[i-1])
            if i>0:
                T_i_mid = (T_i + T[i-1])/2
            else:
                T_i_mid = T_i/2
            T_avg_long += T_i_mid*dR
        # When T_i < T_thresh
        else:
            R_val= R[i]
            dR = np.amax([0.0,R_val-R_prev])
            if i>0:
                T_i_mid = (T_i + T[i-1])/2
            else:
                T_i_mid = T_i/2
            T_avg_short += T_i_mid*dR
        # T_avg += T_i*dR/R_tot
        if R_val > R_prev:
            R_prev = R_val
    T_avg_total = (T_avg_short + T_avg_long)/R_tot
    T_avg_short = T_avg_short / (R_half-R_diff_half)
    T_avg_long = T_avg_long / (R_tot- R_half + R_diff_half)
    return T_avg_total, T_avg_short, T_avg_long, R_half

# def get_T_avg(T,R,R_tot):
#     # R_max = np.amax(R)
#     R_prev = 0.
#     T_avg_total = 0
#     T_avg_long = 0
#     T_thresh = None
#     # numerical threshold for R that defines the long timescale average
#     R_long =  (1-1/np.exp(1))*R_tot
#     dR_arr = []
#     # I added this as an exeption to handle the true model values better (otherwise they are not correctly normalized, because we only compute R(T) up to T = 3sec)
#     if R_tot > np.amax(R):
#         R_tot = np.amax(R)
#     for i, T_i in enumerate(T):
#         # If T_i < T_thresh?
#         if R[i]<R_long:
#             R_val= R[i]
#             dR = np.amax([0.0,R_val-R_prev])
#             dR_arr += [dR]
#             # ensures that this only happens at first threshold crossing, and not when R decreases for high T
#             if T_thresh == None:
#                 if R[i+1]>R_long:
#                     R_thresh =  R[i+1]
#                     T_thresh = T[i+1]
#             if i>0:
#                 T_i_mid = (T_i + T[i-1])/2
#             else:
#                 T_i_mid = T_i/2
#             T_avg_total += T_i_mid*dR
#         # When T_i > T_thresh, also integrate long timescale
#         else:
#             if R[i]>R_tot:
#                 R_val = R_tot
#             else:
#                 R_val= R[i]
#             dR = np.amax([0.0,R_val-R_prev])
#             dR_arr += [dR]
#             if i>0:
#                 T_i_mid = (T_i + T[i-1])/2
#                 T_i_mid_long = (T_i + T[i-1]-T_thresh)/2
#                 T_avg_long += T_i_mid_long*dR
#             else:
#                 T_i_mid = T_i/2
#                 T_thresh = T[i]
#                 R_thresh = R[i]
#             T_avg_total += T_i_mid*dR
#         # T_avg += T_i*dR/R_tot
#         if R_val > R_prev:
#             R_prev = R_val
#     T_avg_total = T_avg_total/R_tot
#     # there are weird cases where the neuron has very high history dependence at one point, which yields an Rtot that is equal to that peak, and which makes it impossible to define a long timescale
#     if (R_tot- R_thresh) == 0:
#         T_avg_long = T_avg_total
#         print(T_avg_total)
#     else:
#         T_avg_long = T_avg_long / (R_tot- R_thresh)
#     return T_avg_total, T_avg_long, R_thresh, T_thresh, dR_arr

def get_running_avg(R):
    R_avg = np.zeros(len(R))
    R_avg[0] = R[0]
    for i in np.arange(1,len(R)-1):
        R_avg[i] = (R[i-1]+R[i]+R[i+1])/3.
    R_avg[-1] = (R[-1]+R[-2])/2
    return R_avg

# Write these functions such that you can handle R and autiMI and Corr similarily:
def get_dR(T,R,R_tot):
    R_prev = 0.
    dR_arr = []
    for i, T_i in enumerate(T):
        # No values higher than R_tot are allowed, otherwise the integrated timescale might be misestimated because of spurious contributions at large T
        if R[i]>R_tot:
            R_val = R_tot
        else:
            R_val= R[i]
        # No negative increments are allowed
        dR = np.amax([0.0,R_val-R_prev])
        dR_arr += [dR]
        # The increment is taken with respect to the highest previous value of R
        if R_val > R_prev:
            R_prev = R_val
    return np.array(dR_arr)

def get_dR_running_average(T,R,R_tot):
    R_prev = 0.
    R_long =  (1-1/np.exp(1))*R_tot
    T_thresh = None
    dR_arr = []
    for i, T_i in enumerate(T):
        # If T_i < T_thresh?
        if R[i]<R_long:
            R_val= R[i]
            dR = np.amax([0.0,R_val-R_prev])
            dR_arr += [dR]
            # ensures that this only happens at first threshold crossing, and not when R decreases for high T
            if T_thresh == None:
                if R[i+1]>R_long:
                    R_thresh =  R[i+1]
                    T_thresh = T[i+1]
        else:
            if R[i]>R_tot:
                R_val = R_tot
            else:
                R_val= R[i]
            dR = np.amax([0.0,R_val-R_prev])
            dR_arr += [dR]
            if i==0:
                T_thresh = T[i]
                R_thresh = R[i]
        if R_val > R_prev:
            R_prev = R_val
    return np.array(dR_arr), R_thresh, T_thresh


def get_dR_in_equal_steps(step_size, N_steps, T, R, R_tot):
    dR_arr = []
    for i in range(N_steps):
        t_lo = i*step_size
        t_hi = (i+1)*step_size
        if t_lo <T[0]:
            T_lo = 0.0
            R_lo = 0.0
        else:
            T_lo = T[T<=t_lo][-1]
            R_lo = np.amax(R[T<=t_lo])
        T_hi = T[T>=t_hi][0]
        R_hi = np.amax(R[T<=T_hi])
        # no values higher than R_tot are allowed to avoid suprious contributions due to statistical fluctuations.
        R_hi = np.amin([R_hi, R_tot])
        dT = T_hi - T_lo
        dR = (R_hi-R_lo)*step_size/dT
        dR_arr += [dR]
    return np.array(dR_arr)

# def get_T_avg(T, dR_arr):
#     T = np.append([0],T)
#     # T_avg for all T
#     norm = 2*np.sum(dR_arr)
#     T_left_sum = np.sum(T[i]*dR for i,dR in enumerate(dR_arr))
#     T_right_sum = np.sum(T[i+1]*dR for  i,dR in enumerate(dR_arr))
#     T_avg = (T_left_sum+T_right_sum)/norm
# #     # Define arrays for long T
# #     T_thresh = T[T>T_avg][0]
# #     T_thresh = T[T>0.01][0]
# #     T_long = T[T>T_thresh]-T_thresh
# #     T_long = np.append([0],T_long)
# #     # add another element to match dimensions
# #     dR_arr_long = np.append([0],dR_arr)[T>T_thresh]
# #     # T_avg but for long T
# #     norm = 2*np.sum(dR_arr_long)
# #     T_left_sum = np.sum(T_long[i]*dR for i,dR in enumerate(dR_arr_long))
# #     T_right_sum = np.sum(T_long[i+1]*dR for  i,dR in enumerate(dR_arr_long))
# #     T_avg_long = (T_left_sum+T_right_sum)/norm
#     return T_avg#, T_avg_long

def get_T_avg(T, dR_arr, T_0):
    T_left = np.append([0],T)[:-1]
    # Only take into considerations contributions beyond T_0
    T_right = T[T_left>=T_0]
    dR_arr = dR_arr[T_left>=T_0]
    T_left = T_left[T_left>=T_0]
    norm = np.sum(dR_arr)
    if norm == 0.:
        T_avg = 0.0
    else:
        T_offset = T_left[0]
        T_left = T_left - T_offset
        T_right = T_right - T_offset
        T_left_sum = np.sum(T_left[i]*dR for i,dR in enumerate(dR_arr))
        T_right_sum = np.sum(T_right[i]*dR for  i,dR in enumerate(dR_arr))
        T_avg = (T_left_sum+T_right_sum)/2/norm
    return T_avg

#     T = np.append([0],T)
#     T_thresh = T[T>=T_0][0]
#     T_shifted = T[T>T_thresh]-T_thresh
#     T_shifted = np.append([0],T_shifted)
#     dR_arr_shifted = np.append([0],dR_arr)[T>T_thresh]
#     norm = 2*np.sum(dR_arr_shifted)



# def get_T90_exponential(T,R,R_tot)
    # scipy.optimize.curve_fit(lambda t,a,b: a*numpy.exp(b*t),  x,  y,  p0=(4, 0.1)
    # scipy.optimize.curve_fit(lambda T,tau: R_tot - (R_tot-R85)*numpy.exp((T85-T)/tau),  T[T85_index:max_vaid_index],  R[T85_index::max_vaid_index], p0 = ((T_D-T85)/5)

    # T90 = T85 - tau*np.log((R_tot-R90)/(R_tot-R85))
    # return T90
