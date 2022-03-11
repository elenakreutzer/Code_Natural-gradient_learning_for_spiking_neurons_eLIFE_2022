The code is written for an hpc slurm cluster

1) Prerequisites
-Set simulation parameters in "File01_simulation_parameters".
-create lookup table by submitting create_lt.sh to the slurm cluster (sbatch create_lt.sh, careful, list of "q" (see Eqn. 36 in paper) in this file might need expansion if different input rates or number of inputs are chosen, list must be always exactly in the same form in create_lt.sh,File04a_merge_lt.py, and File05_main_simulation.py!! )
-sample input spikes by submitting sample_input_spikes to the slurm cluster (sbatch sample input_spikes)
-merge input spikes (python2.7 File03_merge_input_spikes.py)
-merge lookup_table (python2.7 File04a_merge_lt.py) 

2) Simulation learning curves

-submit rate_error.sh to the cluster

3) Merge results learning curves
-Run File06_merge_results.py (python2.7 File06_merge_results.py)

4) Simulation B-C PSTH, Voltage traces
-submit psth.sh to the cluster

5) Merge results PSTHs
-Run File09_merge_psth.py (python2.7 File09_merge_psth.py)

5)Plot
-Run File06a_plot_lc.py (python2.7 File06a_plot_lc.py) (Plots Fig3F and Fig9A-C)
-Run File10_plot_psth.py (File10_plot_psth.py) (plots Fig3B-C)



