# Embedding optimization reveals long-lasting history dependence in neural spiking activity

These files accompany the results obtained in "Embedding optimization reveals long-lasting history dependence in neural spiking activity". For further details please refer to the manuscript posted on [bioarxiv](https://www.biorxiv.org/content/10.1101/2020.11.05.369306v1).

Most of the analysis code is based on the hdestimator [toolbox](https://github.com/Priesemann-Group/hdestimator).

## Dependencies
- Python (>=3.2)
- h5py
- pyyaml
- numpy
- scipy
- mpmath
- matplotlib
- pyNN
- mrestimator

### Optional Dependencies
- cython, for significantly faster running times

## Installation
Python packages can be installed either via your operating system's package manager or
using eg pip or conda.

* ubuntu: `sudo apt install python3-h5py python3-yaml python3-numpy python3-scipy python3-mpmath python3-matplotlib cython3`

* fedora: `sudo dnf install python3-h5py python3-pyyaml python3-numpy python3-scipy python3-mpmath python3-matplotlib python3-Cython`

* using pip: `pip install h5py pyyaml numpy scipy mpmath matplotlib cython3`

* using conda: `conda install h5py pyyaml numpy scipy mpmath matplotlib cython`

### Recommended: Compile the Cython modules

From within the repository's base folder, change into the `src` directory:

`cd src`

There, compile the Cython modules:

`python3 setup.py build_ext --inplace`

If no errors occured (warnings are OK), you are all set.

### Windows users

Under Windows, you can use the tool eg through miniconda.

Install [miniconda for python3, 32bit](https://docs.conda.io/en/latest/miniconda.html).

To meet the dependencies to compile the Cython modules, download and install [Visual Studio](https://visualstudio.microsoft.com/downloads/).
There, select Desktop development with C++ and install
* MSVC v140 - VS 2015 C++ build tools (v14.00) (more recent versions probably work, too)
* Windows 10 SDK (10.0.18362.0)

Then compile the modules by running the commands as above.

## How to recreate plots

Here, we describe how to reproduce the figures of the main paper, once the analysis results were obtained. See the next section on how to reproduce the analysis results.

### Figure 1

To plot results in Fig 1C and 1D run `python plots/fig1/plot_experimental_examples.py`.

### Figure 3

To plot results in Fig 3B run `python plots/fig3/binary_AR.py vs_h`.

To plot results in Fig 3C for l=1 run `python plots/fig3/binary_AR.py vs_d l=1`, and replace `l=1` with `l=5` to plot results for l=5.

To plot results in Fig 3D run `python plots/fig3/binary_AR.py vs_m`.

To plot results in Fig 3E run `python plots/fig3/binary_AR.py vs_l`.

### Figure 4

To recreate Fig 4A run `python plots/fig4/plot_branching_process_activity.py`, and to plot results in Fig 4B run `python plots/fig4/plot_branching_process_analysis.py`.

To recreate Fig 4C first change directory with `cd plots/fig4/` and then run `python plot_izhikevich_dynamics.py nest --plot-figure`. This will create a folder `Results` where you can find the plot. To plot results in Fig 4D, run
`python plots/fig4/plot_izhikevich_analysis.py`.

To plot the spike train in Fig 4E run `python plots/fig4/plot_glif_spiketrain.py` and to plot results in Fig 4F run `python plots/fig4/plot_glif_analysis.py`.

### Figure 5

To plot results in Fig 5A run `python plots/fig5/plot_R_vs_d_bbc.py`.

To plot results in Fig 5B run `python plots/fig5/plot_R_vs_d_shuffling.py`.

To plot results in Fig 5C run `python plots/fig5/plot_R_vs_T.py` and to plot the zoomin run `python plots/fig5/plot_R_vs_T_zoomin.py`.

### Figure 6

To plot results in Fig 6B run `python plots/fig6/plot_R_vs_T_example.py`.

To plot results in Fig 6C run `python plots/fig6/plot_R_vs_T_relative_to_exponential.py`.

To plot results in Fig 6D run `python plots/fig6/plot_Rtot_relative_to_bbc.py`.

### Figure 7

To plot results in Fig 7A run `python plots/fig7/plot_Rtot_vs_tau_R.py`, and for 7B run `python plots/fig7/plot_Rtot_vs_tau_C.py`.

To recreate the raster plots of the spiking activity, run `python plots/fig7/plot_spiking_activity.py ()`, where `()` should be replaced by `CA1` for rat dorsal hippocampus CA1, `retina` for salamander retina, `culture` for rat cortical culture and `V1` for mouse primary visual cortex.

### Figure 8

To plot results in Fig 8, run `python plots/fig8/plot_R_vs_T.py` to plot the top row of all panels, and `python plots/fig8/plot_autocorrelation_vs_T.py` to plot the bottom row of all panels.

## How to recreate analysis results

Here, we describe how to reproduce the analysis results for given spike trains. See the next section on how you can obtain the spike trains from the published data sets.

### Figure 1

To reproduce the results for $R(T)$ and $\Delta R(T)$ in Fig 1, run `python exe/emb_opt_analysis.py 20 V1 90min example_neuron`.

### Figure 3
To reproduce the results for Fig 3, first run `python exe/binary_AR_analysis.py adapt_m_and_h`, and then
`python exe/binary_AR_analysis.py vs_m` and
`python exe/binary_AR_analysis.py vs_l`.

### Figure 4
To generate the spike trains for the different models run
- branching process: `python exe/branching_process_simulation.py 90min`
- izhikevich neuron: `python exe/izhikevich_simulation.py`
- generalized leaky integrate-and-fire (GLIF) neuron with spike adaptation: `python exe/glif_simulation.py 0 glif_22s_kernel 900min`

To analyze the branching process and to reproduce the results in Fig 4B, run `python exe/branching_process_lagged_MI_analysis.py` and `python exe/emb_opt_analysis.py 0 branching_process 90min full_shuffling`.

To analyze the izhikevich neuron and to reproduce the results in Fig 4D, run `python exe/izhikevich_lagged_MI_analysis.py` and `python exe/emb_opt_analysis.py 0 izhikevich_neuron 20min full_bbc`.

To analyze the GLIF neuron and to reproduce the results in Fig 4E, run `python exe/glif_lagged_MI_analysis.py` and `python exe/glm_emb_opt.py [index] glif_22s_kernel` where `[index]` = `1,...,50`.

### Figure 5

To reproduce the results shown in Fig 5A and 5B, run `python exe/estimation_R_vs_d.py [sample_index]`, where `[sample_index]` = `1,...,50`. To obtain the ground truth by fitting a GLM, run `python exe/glm_emb_benchmark_R_vs_d.py 0 glif_1s_kernel 900min`.

To reproduce results shown in Fig 5C, run `python exe/emb_opt_analysis.py 4 glif_1s_kernel 90min full_bbc` and `python exe/emb_opt_analysis.py 4 glif_1s_kernel 90min full_shuffling`. To compute the confidence intervals, run  `python exe/confidence_intervals.py 4 glif_1s_kernel 90min [setup]` for the same setups as before. To compute the ground truth by fitting a GLM, run `python exe/glm_emb_opt.py [index] glif_1s_kernel` where `[index]` = `1,...,40`.

### Figure 6

To reproduce the results shown in Fig 6B, run `python exe/emb_opt_analysis.py 10 CA1 90min [setup]`, where you replace `[setup]` with `full_bbc`, `full_shuffling`, `fivebins`, `onebin`. When this was completed, run `python exe/glm_emb_opt.py 10 CA1` to compute the GLM estimate.
To compute the confidence intervals, run `python exe/confidence_intervals.py 10 CA1 90min [setup]` with the same setups as above.

To reproduce results in Fig 6D, repeat the above steps and run `python exe/emb_opt_analysis.py [neuron_index] [neural_system] 90min [setup]` for all setups `full_bbc`, `full_shuffling`, `fivebins`, `onebin` and the following neural systems and neuron indices:
| `[neural_system]`   | `[neuron_index]` |
| :------------- | :----------: |
|  `CA1` | `1,...,28`  |
| `retina`  |  `1,...,111` |
| `culture` |  `1,...,48`  |

To additionally reproduce results in Fig 6C, run `python exe/emb_opt_analysis.py [neuron_index] CA1 90min [setup]` with `[neuron_index]` = `1,...,28` and the setups `full_bbc_uniform`, `full_shuffling_uniform` and `fivebins_uniform`.

### Figure 7

To reproduce results in Fig 7A, run `python exe/emb_opt_analysis.py [neuron_index] [neural_system] 40min fivebins` for the same neural systems and neuron indices as for Fig 6, and additionally for `[neural_system]` = `V1` with `[neuron_index]` = `1,...,142`.

To additionally reproduce results in Fig 7B, run `python exe/auto_correlation_analysis.py 10`.

### Figure 8

If you have run the analysis for Fig 7, you can skip the following step: Run `python exe/emb_opt_analysis.py [neuron_index] V1 40min fivebins` with `[neuron_index]` = `20`, `1` ,`30`.

To obtain the confidence intervals in the top row, additionally run `python exe/confidence_intervals.py [neuron_index] V1 40min fivebins` with `[neuron_index]` = `20`, `1` ,`30`.

## How to preprocess experimental data

Here, we describe how to preprocess the experimental data to obtain the spike times of individual sorted units that are required for the analysis.

**Rat dorsal hippocampus (CA1)** Download the file `ec014.277.spike_ch.mat` from http://crcns.org/data-sets/hc/hc-2, and copy it to `data/CA1/raw/`. Then run `python exe/preprocess_data.py CA1`. This will create files that contain the spike times of individual sorted units in `data/CA1/spks/`, as well as a list `validNeurons.npy` that contains the indices of the sorted units that passed the firing rate criterion ($ 0.5 \,\text{ms} \leq \text{rate} \leq 10 \,\text{ms}$).

**Salamander retina** Download the file `mode_paper_data/unique_natural_movie/data.mat` from https://datadryad.org/stash/dataset/doi:10.5061/dryad.1f1rc, and copy it to `data/retina/raw/mode_paper_data/unique_natural_movie/`. Then run `python exe/preprocess_data.py retina`. This will create files that contain the spike times of individual sorted units in `data/retina/spks/`, as well as a list `validNeurons.npy` that contains the indices of the sorted units that passed the firing rate criterion ($ 0.5 \,\text{ms} \leq \text{rate} \leq 10 \,\text{ms}$).

**Rat cortical culture** Download the files `L_Prg035_txt_nounstim.txt`, `L_Prg036_txt_nounstim.txt`,`L_Prg037_txt_nounstim.txt` ,`L_Prg038_txt_nounstim.txt` ,`L_Prg039_txt_nounstim.txt` from https://data.mendeley.com/datasets/4ztc7yxngf/1, and copy them to `data/culture/raw/`. Then run `python exe/preprocess_data.py culture`. This will create files that contain the spike times of individual sorted units in `data/retina/spks/`, as well as a list `validNeurons.npy` that contains the indices of the sorted units that passed the firing rate criterion ($ 0.5 \,\text{ms} \leq \text{rate} \leq 10 \,\text{ms}$).

**Mouse primary visual cortex** Download the data archive from https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750, and extract it to `data/neuropixels/raw/`. Then run `python exe/preprocess_data.py V1`. This will create files that contain the spike times of individual sorted units in `data/neuropixels/Waksman/V1/spks/`, as well as a list `validNeurons.npy` that contains the indices of the sorted units that passed the firing rate criterion ($ 0.5 \,\text{ms} \leq \text{rate} \leq 10 \,\text{ms}$).
