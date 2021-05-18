from sys import stderr, exit, argv
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np

sample_index = int(argv[1])
rec_length = argv[2]

if len(argv) > 3:
    data_path = argv[3]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

DATA_DIR = '{}/glif_1s_kernel'.format(data_path)

rec_lengths = {'1min': 60., '3min': 180., '5min': 300.,
               '10min': 600., '20min': 1200., '45min': 2700., '90min': 5400., '300min':18000.}

T_rec = rec_lengths[rec_length]
# When recording longer than 30 minutes, Shift by 90 minutes to obtain 10 representative samples with different statstics. Otherwise, shift by 30minutes to obtain 30 samples.
# if T_rec > 1800.:
#     t_0 = 100. - 10**(-4) + sample_index * 5400.
# else:
#     t_0 = 100. - 10**(-4) + sample_index * 1800.
t_0 = 100. - 10**(-4)

spiketimes = np.load('{}/spiketimes_90min_{}.npy'.format(DATA_DIR, sample_index))
spiketimes = spiketimes - t_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
