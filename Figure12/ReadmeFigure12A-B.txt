Code is written for SLURM HPC CPU Cluster

1) Set rates in File01_parameter_angle.py
2) Submit create_lt_c.sh to the cluster (sbatch create_lt_c.sh)
3) Run File03_merge_lt_c.py (python2.7 File03_merge_lt_c.py)
4) Submit create_lt.sh to the cluster (sbatch create_lt.sh)
5) Run File04a_merge_lt.py (python2.7 File04a_merge_lt.py)
6) Submit angles_euc.sh to the cluster (sbatch angles_euc.sh)
7) Run File06_merge_euc_angles (python2.7 File06_merge_euc_angles.py)
8) Submit angles_fisher.sh to the cluster (sbatch angles_fisher.sh)
9) Run File08_merge_fisher_angles (python2.7 File08_merge_fisher_angles.py)
10)Repeat 1)-9) for all desired input rate combinations
11) Plot with File09_plot_angles.py (python2.7 File09_plot_angles.py)

2