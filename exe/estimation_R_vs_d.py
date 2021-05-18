import sys
from sys import argv, path
import os
from os.path import realpath, dirname
from subprocess import call
from scipy.optimize import bisect
import numpy as np
import time
from collections import Counter

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))

if 'hde_glm' not in sys.modules:
    import hde_glm as glm
    import hde_plotutils as plots
    import hde_api as hapi
    import hde_embedding as emb
    import hde_utils as utl
    import hde_fast_embedding as fast_emb
    from hde_fast_glm import *
    import estimators_lucas_implementation as est

device_or_sample_index = argv[1]

if device_or_sample_index == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    sample_index = int(device_or_sample_index)

recorded_system = 'glif_1s_kernel'
rec_length = '90min'
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)
ANALYSIS_DIR = '{}/analysis/{}/analysis_R_vs_d'.format(CODE_DIR, recorded_system)
t_0 = 100-10**(-5)

"""Parameters"""
FAST_EMBEDDING_AVAILABLE = True
embedding_step_size = 0.005
Trec = 5400
tau = 0.02
past_range_T = 1.0
kappa = 0.0
N_d = 30
d_max= 60
d_min= 1
kappa_d = bisect(lambda kappa: np.sum([d_min*np.power(10,i*kappa) for i in range(N_d)])-d_max,0,1.)
d=0
d_list=[]
for i in range(N_d):
    d+=np.power(10,kappa_d*i)
    d_list+=[d]
embedding_number_of_bins_set = np.array(d_list).astype(int)
np.save('{}/embedding_number_of_bins_set.npy'.format(ANALYSIS_DIR),embedding_number_of_bins_set)

print(embedding_number_of_bins_set)
R_plugin_list = []
R_nsb_list = []
bbc_term_list = []
R_shuffling_list = []
R_shuffling_correction_list = []
# Take out the last
for j, number_of_bins_d in enumerate(embedding_number_of_bins_set[:-1]):
    past_range_T = number_of_bins_d*tau
    embedding = (past_range_T, number_of_bins_d, kappa)
    spiketimes = np.load('%s/spiketimes_%s_%d.npy'%(DATA_DIR, rec_length, sample_index))
    # spiketimes = np.load('%s/spiketimes_900min.npy'%(DATA_DIR))
    spiketimes = spiketimes - t_0
    spiketimes = spiketimes[spiketimes>0]
    spiketimes = spiketimes[spiketimes<Trec]
    # time1 = time.clock_gettime(1)
    # symbol_counts = utl.add_up_dicts([Counter(fast_emb.get_symbol_counts(spike_times, embedding, embedding_step_size)) for spike_times in [spiketimes]])
    counts = counts_C(spiketimes, embedding_step_size, 0, Trec, 'medians')
    past_array = past_activity_array(spiketimes, counts, number_of_bins_d ,kappa , tau , embedding_step_size, 0 , Trec , 'medians')
    # time2 = time.clock_gettime(1)
    # print("embeddings done in in", (time2-time1), "seconds")
    # time1 = time.clock_gettime(1)
    R_nsb, R_plugin,  R_shuffling, R_shuffling_correction = est.get_estimates(counts, past_array,  number_of_bins_d, estimators='all', mode='medians', correction_out=True)
    bbc_term = abs(R_plugin - R_nsb)/R_nsb
    # time2 = time.clock_gettime(1)
    # print("lucas estimates done in", (time2-time1), "seconds")
    # R_plugin_daniel = hapi.get_history_dependence('plugin', symbol_counts, number_of_bins_d)
    # time1 = time.clock_gettime(1)
    # R_bbc, bbc_term = hapi.get_history_dependence('bbc', symbol_counts, number_of_bins_d)
    # time2 = time.clock_gettime(1)
    # print("nsb done in in", (time2-time1), "seconds")
    # time1 = time.clock_gettime(1)
    # R_shuffling, shuffling_correction = hapi.get_history_dependence('shuffling_with_correction', symbol_counts, number_of_bins_d)
    # time2 = time.clock_gettime(1)
    # print("nsb done in in", (time2-time1), "seconds")
    # print(number_of_bins_d, R_nsb,R_plugin, R_shuffling, bbc_term)
    print(number_of_bins_d, R_nsb, R_plugin)
    R_plugin_list += [R_plugin]
    R_nsb_list += [R_nsb]
    bbc_term_list += [bbc_term]
    R_shuffling_list += [R_shuffling]
    R_shuffling_correction_list += [R_shuffling_correction]

np.save('%s/R_plugin_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), R_plugin_list)
np.save('%s/R_nsb_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), R_nsb_list)
np.save('%s/bbc_term_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), bbc_term_list)
np.save('%s/R_shuffling_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), R_shuffling_list)
np.save('%s/R_shuffling_correction_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), R_shuffling_correction_list)
