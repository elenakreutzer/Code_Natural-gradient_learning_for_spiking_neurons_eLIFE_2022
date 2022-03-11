The code is written for a hpc slurm cluster

Fig3DE consist of 3 different parts

-vectorplot
-costfunction
-weightpath

1) Prerequisites
-create lookup table by submitting create_lt.sh to the slurm cluster (sbatch creat_lt.sh)
-sample input spikes by submitting sample_input_spikes to the slurm cluster (sbatch sample input_spikes)
-merge input spikes (python2.7 File03_merge_input_spikes.py)

2) Vectorplots
-submit vec_nat.sh to the cluster for natural gradient vectors (sbatch vec_nat.sh)
-submit vec_euc.sh to the cluster for Euclidean gradient vectors (sbatch vec_euc.sh)

3)Cost function
-submit cost.sh to the cluster (sbatch cost.sh)

4) Weightpaths
-submit rate_error.sh to the cluster (sbatch rate_error.sh, same code as for 3F but with n=2 (set in "File01_simulation_parameters.py").

5) Merge results
-merge vectors and cost (python2.7 File05_merge_vectorplots.py)
-merge weightpaths (python2.7 File07_merge_wpath.py)

6)Plot
-Run File07a_plot_wpath.py (File07a_plot_wpath.py)




