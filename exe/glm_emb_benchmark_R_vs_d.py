import os
import sys
from sys import exit, argv, path
from os.path import realpath, dirname
import pandas as pd
import yaml
import numpy as np
from scipy.optimize import bisect
import csv

# CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
CODE_DIR = '/home/lucas/research/projects/history_dependence/historydependence'
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in sys.modules:
    import hde_glm as glm
    import hde_plotutils as plots

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

"""Run parameters"""

device_or_run_index = argv[1]
recorded_system = argv[2]
rec_length = argv[3]
ANALYSIS_DIR = '{}/analysis/{}/analysis_R_vs_d'.format(CODE_DIR, recorded_system)
# d_list = np.load('{}/embedding_number_of_bins_set.npy'.format(ANALYSIS_DIR)).astype(int)
tau = 0.02
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
d_list = np.array(d_list).astype(int)
N_d = len(d_list)
T_list = d_list * tau
kappa_list = np.zeros(N_d)
tau_list = np.zeros(N_d)+ tau
# Get embedding parameters from embedding optimization
embedding_parameters = np.array([T_list, d_list, kappa_list, tau_list])

if len(argv) > 4:
    data_path = argv[4]
else:
    data_path = '{}/data'.format(CODE_DIR)

if device_or_run_index == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

if device_or_run_index == 'cluster':
    sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    sample_index = int(device_or_run_index)

def main():
    # Load glm settings
    with open('{}/settings/{}_glm.yaml'.format(CODE_DIR,recorded_system), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)

    # Load the 900 minute simulated recording
    DATA_DIR = '{}/{}'.format(data_path,recorded_system)
    spiketimes = np.load('{}/spiketimes_{}.npy'.format(DATA_DIR, rec_length))

    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(spiketimes, glm_settings)

    # Compute history dependence with GLM for the same embeddings as found with bbc/shuffling
    glm_benchmark = glm.compute_benchmark(embedding_parameters, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm_csv_file_name = '{}/glm_benchmark_{}_tau{}.csv'.format(
        ANALYSIS_DIR, rec_length, int(tau*1000))
    with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM"])
        writer.writeheader()
        for i, T in enumerate(embedding_parameters[0]):
            writer.writerow(
                {"T": T, "number_of_bins_d": int(embedding_parameters[1][i]), "scaling_kappa": embedding_parameters[2][i], "first_bin_size": embedding_parameters[3][i], "R_GLM": glm_benchmark[i]})
    return EXIT_SUCCESS


if __name__ == "__main__":
    exit(main())
