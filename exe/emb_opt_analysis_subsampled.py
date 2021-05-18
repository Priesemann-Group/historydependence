from sys import argv
import os
from os.path import realpath, dirname
from subprocess import call

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))

"""Run parameters"""
device_or_run_index = argv[1]
recorded_system = argv[2]
rec_length = argv[3]
setup = argv[4]

if device_or_run_index == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    run_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    run_index = int(device_or_run_index)

load_script = '{}/exe/load_spiketimes_subsampled.py'.format(
    CODE_DIR, recorded_system)

setting_file = '{}/settings/{}_{}_subsampled.yaml'.format(
    CODE_DIR, recorded_system, setup)

program = '/home/lucas/anaconda2/envs/python3/bin/python'
# program='/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (CODE_DIR)
# if spiketimes are stored at a custom location, this can be passed to the script
if len(argv) > 5:
    data_path = argv[5]
    load_arguments = ' ' + str(run_index) + ' ' + rec_length + ' ' + data_path
else:
    load_arguments = ' ' + str(run_index) + ' ' + rec_length

"""Compute estimates for different embeddings"""

command = program + ' ' + load_script + load_arguments + ' | ' + program + ' ' + script + ' /dev/stdin -t hist -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length, setup, str(run_index))

call(command, shell=True)
print("hist done")

"""Compute essential confidence intervals"""

command = program + ' ' + load_script + load_arguments + ' | ' + program + ' ' + script + ' /dev/stdin -t conf -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length, setup, str(run_index))

call(command, shell=True)
print("conf done")
"""Create csv results files"""

command = program + ' ' + load_script + load_arguments + ' | ' + program + ' ' + script + ' /dev/stdin -t csv -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length, setup, str(run_index))

call(command, shell=True)
print("csv done")
"""Create plots"""

command = program + ' ' + load_script + load_arguments + ' | ' + program + ' ' + script + ' /dev/stdin -t plots -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length, setup, str(run_index))

call(command, shell=True)
