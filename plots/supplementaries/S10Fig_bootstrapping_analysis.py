from sys import exit, path
import yaml

with open('settings.yaml', 'r') as settings_file:
    settings = yaml.load(settings_file, Loader=yaml.BaseLoader)

path.insert(1, settings['estimator_src_dir'])

import hde_utils as utl
import hde_api as hapi
import hde_visualization as vsl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py

### settings

data_dir = settings['data_dir']

spike_times_full_file_name = "spiketimes_constI_5ms.dat"

estimation_method = "bbc"
embedding_step_size = 0.005

number_of_bootstraps = 250
number_of_CIs = 100

analysis_dir = 'analysis_files'
analysis_file_name = 'bs_analysis_{}.h5'.format(estimation_method)

target_recording_lengths_in_min = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20, 25]

# optimization algorithm run independently on 5 datasets of the given recording length
# the representative embedding is that which yielded the median max R
# (bbc estimator)
representative_embeddings = {1: (0.28117, 3, 0.0),
                             2: (0.25059, 4, 0.11448),
                             3: (0.44563, 6, 0.14992),
                             4: (0.44563, 6, 0.03748),
                             5: (0.31548, 6, 0.06693),
                             7: (0.44563, 6, 0.03748),
                             9: (0.35397, 8, 0.06837),
                             11: (0.35397, 9, 0.05739),
                             13: (0.56101, 8, 0.08019),
                             15: (0.5, 8, 0.05151),
                             20: (0.44563, 10, 0.03587),
                             25: (0.62946, 10, 0.08108)}

axes_fontsize = 12


###

def get_bs_data(analysis_file,
                spike_times_full,
                number_of_bootstraps,
                embedding,
                target_recording_length_in_min,
                number_of_recordings,
                target_data="std"):
    assert target_data in ["std", "CI"]
    
    # if there is useful data available, use that
    h5_data_parent_dir_name = "{}/{}".format(target_recording_length_in_min,
                                             str(embedding))
    if h5_data_parent_dir_name in analysis_file:
        stored_number_of_bootstraps \
            = max([int(k) for k in analysis_file[h5_data_parent_dir_name].keys()])
        if not stored_number_of_bootstraps >= number_of_bootstraps:
            stored_number_of_bootstraps = number_of_bootstraps
    else:
        stored_number_of_bootstraps = number_of_bootstraps    
    
    h5_data_dir_name = "{}/{}/{}".format(target_recording_length_in_min,
                                         str(embedding),
                                         stored_number_of_bootstraps)

    if not h5_data_dir_name in analysis_file:
        analysis_file.create_group(h5_data_dir_name)
        
    h5_data_dir = analysis_file[h5_data_dir_name]

    bs_data = []

    for rep in range(number_of_recordings):
        if not str(rep) in h5_data_dir:
            bs_spike_times = get_random_chunk_of_data(spike_times_full,
                                                      60 * target_recording_length_in_min)
            bs_recording_length = bs_spike_times[-1] - bs_spike_times[0]

            bs_firing_rate = utl.get_binned_firing_rate(bs_spike_times, embedding_step_size)
            block_length_l = max(1, int(1 / (bs_firing_rate * embedding_step_size)))

            bs_Rs = utl.get_bootstrap_history_dependence([bs_spike_times],
                                                         embedding,
                                                         embedding_step_size,
                                                         estimation_method,
                                                         stored_number_of_bootstraps,
                                                         block_length_l)

            R = hapi.get_history_dependence_for_single_embedding(bs_spike_times,
                                                                 bs_recording_length,
                                                                 estimation_method,
                                                                 embedding,
                                                                 embedding_step_size,
                                                                 bbc_tolerance=np.inf)

            d = h5_data_dir.create_group(str(rep))
            d.create_dataset("bs_history_dependence", data=bs_Rs)
            d.create_dataset("history_dependence", data=R)

        bs_Rs = h5_data_dir["{}/bs_history_dependence".format(rep)][()][:number_of_bootstraps]
        R = h5_data_dir["{}/history_dependence".format(rep)][()]

        if target_data == "std":
            bs_data += [np.std(bs_Rs)]
        elif target_data == "CI":
            sigma_R = np.std(bs_Rs)
            CI_lo = R - 2 * sigma_R
            CI_hi = R + 2 * sigma_R

            bs_data += [(CI_lo, CI_hi)]
        
    return bs_data


def get_bs_histdep_on_full(analysis_file,
                           spike_times_full_file_name,
                           embedding,
                           number_of_bootstraps,
                           target_recording_length_in_min):

    h5_data_dir_name = "{}/{}/{}".format(spike_times_full_file_name,
                                         target_recording_length_in_min,
                                         str(embedding))

    if not h5_data_dir_name in analysis_file:
        analysis_file.create_group(h5_data_dir_name)
    h5_data_dir = analysis_file[h5_data_dir_name]

    if "bs_history_dependence" in h5_data_dir:
        stored_bs_history_dependence = h5_data_dir["bs_history_dependence"][()]
    else:
        stored_bs_history_dependence = []

    if len(stored_bs_history_dependence) >= number_of_bootstraps:
        return stored_bs_history_dependence[:number_of_bootstraps]

    else:
        spike_times_full \
            = utl.get_spike_times_from_file("{}/{}".format(data_dir,
                                                           spike_times_full_file_name))[0]
        
        bs_Rs = np.zeros(number_of_bootstraps - len(stored_bs_history_dependence))
        for rep in range(len(bs_Rs)):
            bs_spike_times = get_random_chunk_of_data(spike_times_full,
                                                      60 * target_recording_length_in_min)

            bs_recording_length = bs_spike_times[-1] - bs_spike_times[0]

            bs_R = hapi.get_history_dependence_for_single_embedding(bs_spike_times,
                                                                    bs_recording_length,
                                                                    estimation_method,
                                                                    embedding,
                                                                    embedding_step_size,
                                                                    bbc_tolerance=np.inf)
            bs_Rs[rep] = bs_R

        bs_history_dependence \
            = np.hstack((stored_bs_history_dependence,
                         bs_Rs))
        
        if len(stored_bs_history_dependence) > 0:
            del h5_data_dir["bs_history_dependence"]
        h5_data_dir.create_dataset("bs_history_dependence", data=bs_history_dependence)

        return bs_history_dependence

def get_random_chunk_of_data(spike_times_full,
                             target_recording_length):
    recording_length_full = spike_times_full[-1] - spike_times_full[0]

    start_time = 100 + (recording_length_full - 200 - target_recording_length) * np.random.random()
    end_time = start_time + target_recording_length

    bs_spike_times = spike_times_full[(spike_times_full >= start_time) &
                                      (spike_times_full < end_time)]
    bs_spike_times -= bs_spike_times[0]

    return bs_spike_times

def compare_R_variances_recording_length(analysis_file, ax):
    spike_times_full \
        = utl.get_spike_times_from_file("{}/{}".format(data_dir,
                                                       spike_times_full_file_name))[0]
    
    recording_length_full = spike_times_full[-1] - spike_times_full[0]
    
    bs_std_means = []
    bs_std_stds = []
    full_stds = []

    for target_recording_length_in_min in target_recording_lengths_in_min:
        print(target_recording_length_in_min)

        embedding = representative_embeddings[target_recording_length_in_min]

        bs_stds = get_bs_data(analysis_file,
                              spike_times_full,
                              number_of_bootstraps,
                              embedding,
                              target_recording_length_in_min,
                              number_of_CIs,
                              target_data="std")


        bs_std_means += [np.average(bs_stds)]
        bs_std_stds += [np.std(bs_stds)]

        # now compute from long recording..
        bs_history_dependence_full = get_bs_histdep_on_full(analysis_file,
                                                            spike_times_full_file_name,
                                                            embedding,
                                                            number_of_bootstraps,
                                                            target_recording_length_in_min)

        full_stds += [np.std(bs_history_dependence_full)]

    ax.errorbar(target_recording_lengths_in_min, bs_std_means, yerr=bs_std_stds,
                color='b', ecolor='b', ls='-')
    ax.plot(target_recording_lengths_in_min, full_stds, color='k', ls='-')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(vsl.format_x_label))

def compare_CIs(analysis_file, ax):
    spike_times_full \
        = utl.get_spike_times_from_file("{}/{}".format(data_dir,
                                                       spike_times_full_file_name))[0]
    recording_length_full = spike_times_full[-1] - spike_times_full[0]

    coverage = []
    
    for target_recording_length_in_min in target_recording_lengths_in_min:
        print(target_recording_length_in_min)

        embedding = representative_embeddings[target_recording_length_in_min]

        # first compute from long recording..
        # get the "true" R for the given embedding
        history_dependence_full \
            = hapi.get_history_dependence_for_single_embedding(spike_times_full,
                                                               recording_length_full,
                                                               estimation_method,
                                                               embedding,
                                                               embedding_step_size,
                                                               bbc_tolerance=np.inf)

        # now compute 95% CIs many times and
        # see whether "true" R was included in the CI 95% of the times
        bs_CIs = get_bs_data(analysis_file,
                             spike_times_full,
                             number_of_bootstraps,
                             embedding,
                             target_recording_length_in_min,
                             number_of_CIs,
                             target_data="CI")

        coverage_this_rec_len = 0
        for CI_lo, CI_hi in bs_CIs:
            if CI_lo <= history_dependence_full <= CI_hi:
                coverage_this_rec_len += 1
        coverage_this_rec_len = 100 * (coverage_this_rec_len / number_of_CIs)
        coverage += [coverage_this_rec_len]


    ax.plot(target_recording_lengths_in_min, coverage, color='g', ls='-')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(vsl.format_x_label))

    
if __name__ == "__main__":
    analysis_file = h5py.File("{}/{}".format(analysis_dir,
                                             analysis_file_name), 'a')
    
    fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, wspace=0.3, hspace=0.7)

    vsl.make_plot_pretty(ax1)
    vsl.make_plot_pretty(ax2)

    ax1.set_ylim([-0.0001, 0.0125])
    ax2.set_ylim([69, 100.1])

    ax1.set_ylabel('standard deviation $\sigma(R)$', fontsize=axes_fontsize)
    ax1.set_xlabel('recording length (min)', fontsize=axes_fontsize)

    ax2.set_ylabel('CI accuracy [%]', fontsize=axes_fontsize)
    ax2.set_xlabel('recording length (min)', fontsize=axes_fontsize)
    
    # ---------------------------------------------------------
    # variance based on many recordings of a given length vs
    # variance based on bootstrapping
    # ---------------------------------------------------------
    #
    # on x: recording length
    # on y: std of estimates
    compare_R_variances_recording_length(analysis_file, ax1)

    # ---------------------------------------------------------
    # compare recording lengths: how good is the 95  
    # ---------------------------------------------------------
    compare_CIs(analysis_file, ax2)

    analysis_file.close()
        
    plt.savefig("CI_benchmark.pdf",
                bbox_inches='tight', dpi=300)
