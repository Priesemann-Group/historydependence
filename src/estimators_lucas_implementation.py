from hde_fast_glm import *
import nsb_estimator as nsb

def _H_plugin(cts):
    cts_local = np.array(cts)
    N = np.sum(cts)
    return -sum(cts_local / float(N) * np.log(cts_local / float(N)))


def _AIS_plugin(cts_joint, cts_marginal, h_spike):
    H_joint = _H_plugin(cts_joint)
    H_marginal = _H_plugin(cts_marginal)
    AIS = h_spike + H_marginal - H_joint
    return AIS, H_joint

def _H_ind(P_spike, P_sp, P_nosp):
    H_sp = -np.sum(P_sp[P_sp > 0] * np.log(P_sp[P_sp > 0])
                   ) - np.sum((1 - P_sp) * np.log(1 - P_sp))
    H_nosp = -np.sum(P_nosp[P_nosp > 0] * np.log(P_nosp[P_nosp > 0])
                     ) - np.sum((1 - P_nosp) * np.log(1 - P_nosp))
    return P_spike * H_sp + (1 - P_spike) * H_nosp


def _AIS_NSB(K_marginal, cts_joint, cts_marginal, h_spike):
    N_trials = np.sum(cts_joint)
    K_joint = K_marginal * 2
    mk_joint = nsb.make_mk(cts_joint, K_joint)
    H_joint = nsb.H_Diri(mk_joint, K_joint, N_trials)
    mk_marginal = nsb.make_mk(cts_marginal, K_marginal)
    H_marginal = nsb.H_Diri(mk_marginal, K_marginal, N_trials)
    AIS = h_spike + H_marginal - H_joint
    # AIS_std=np.sqrt(H_joint[2]**2+H_marginal[2]**2)
    return AIS, H_joint  # AIS_std,

# function to compute all estimates of R.


def get_estimates(counts, past, d, estimators='all', mode='medians', correction_out=False):
    K_past = 2**d
    N_bins = len(counts)
    # spike entropy
    P_spike = np.sum(counts) / float(N_bins)
    H_spike = -P_spike * np.log(P_spike) - \
        (1 - P_spike) * np.log(1 - P_spike)
    # Conventional
    marginal_dat = compress_past(past, d)
    joint_dat = marginal_dat + counts
    cts_joint = make_cts_C(joint_dat)
    cts_marginal = make_cts_C(marginal_dat)
    if estimators == 'all' or estimators == 'NSB':
        R_NSB, H_NSB = _AIS_NSB(K_past, cts_joint,
                                cts_marginal, H_spike) / H_spike
    if estimators != 'NSB':
        R_plugin, H_plugin = _AIS_plugin(
            cts_joint, cts_marginal, H_spike) / H_spike
    if estimators == 'all' or estimators == 'Shuffling':
        marginal_shuffled_dat, P_sp, P_nosp = shuffle_past(
            past, counts, d)
        joint_shuffled_dat = marginal_shuffled_dat + counts
        cts_shuffled_joint = make_cts_C(joint_shuffled_dat)
        cts_shuffled_marginal = make_cts_C(marginal_shuffled_dat)
        H_shuffled_plugin = _H_plugin(cts_shuffled_joint) - H_spike
        # H_shuffled_PT = _H_PT(
        #     N_bins, counts, marginal_shuffled_dat, cts_shuffled_joint) - H_spike
        H_ind = _H_ind(P_spike, P_sp, P_nosp)
        R_shuffled_correction = (H_ind - H_shuffled_plugin) / H_spike
        R_shuffled_plugin = R_plugin - R_shuffled_correction

        # M_shuffled_PT = M_PT - (H_ind - H_shuffled_PT) / H_spike
    if estimators == 'NSB':
        if BBC == True:
            return R_NSB, H_NSB
        else:
            return R_NSB
    if estimators == 'Plugin':
        if BBC == True:
            return R_plugin, H_plugin
        else:
            return R_plugin
    if estimators == 'Shuffling':
        if correction_out == True:
            return R_shuffled_plugin, R_shuffled_correction
        else:
            return R_shuffled_plugin
    if estimators == 'all':
        if correction_out == True:
            return R_NSB, R_plugin,  R_shuffled_plugin, R_shuffled_correction
        else:
            return R_NSB, R_plugin,  R_shuffled_plugin


def _bootstraps(counts, past, N_samples, N_bins, N_bootstraps, d, sampling, estimators='all', mode='medians'):
    # subsample
    R_samples = []
    if sampling == 'consecutive':
        i = 0
        while i < N_bootstraps and (i + 1) * N_samples <= N_bins:
            sample_indices = np.arange(i * N_samples, (i + 1) * N_samples)
            i += 1
            past_subsampled = past[:, sample_indices]
            counts_subsampled = counts[sample_indices]
            R = _estimates(
                counts_subsampled, past_subsampled, d, estimators, mode)
            R_samples += [R]
    else:
        for i in range(N_bootstraps):
            sample_indices = np.random.choice(
                N_bins, size=N_samples, replace=True)
            past_subsampled = past[:, sample_indices]
            counts_subsampled = counts[sample_indices]
            R = _estimates(
                counts_subsampled, past_subsampled, d, estimators, mode)
            R_samples += [R]
    if estimators == 'all':
        return R_samples[:, 0], R_samples[:, 1], R_samples[:, 2]
    else:
        return R_samples
