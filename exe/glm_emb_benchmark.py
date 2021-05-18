import os
import sys
from sys import exit, argv, path
from os.path import realpath, dirname
import pandas as pd
import yaml
import numpy as np

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
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
# rec_length = '90min'
setup = argv[4]
# setup = 'full_bbc'
regularization_method = setup.split('_')[1]
if len(argv) > 5:
    data_path = argv[5]
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
    spiketimes = np.load('{}/spiketimes_900min.npy'.format(DATA_DIR))

    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(spiketimes, glm_settings)

    # Load embedding-optimized estimates
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)

# Load embedding parameters from embedding optimization
    embedding_parameters, analysis_num_str = glm.load_embedding_parameters(
        rec_length, sample_index, ANALYSIS_DIR, regularization_method = regularization_method)

    # Compute glm for optimized embedding parameters for temporal depth, only if sample_index = 0 compute for all T
    if sample_index > 0:
        embedding_parameters_benchmark = embedding_parameters[:,T_D_index:max_valid_index]
    else:
        embedding_parameters_benchmark = embedding_parameters

    # Compute history dependence with GLM for the same embeddings as found with bbc/shuffling
    glm_benchmark = glm.compute_benchmark(embedding_parameters_benchmark, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_benchmark_to_CSV(glm_benchmark, embedding_parameters_benchmark,
                                  ANALYSIS_DIR, analysis_num_str, regularization_method=regularization_method)

    return EXIT_SUCCESS


if __name__ == "__main__":
    exit(main())
