#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=kreutzer@pyl.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end

# Job name
#SBATCH --job-name="heatplot"

# Runtime and memory
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G

# Partition
##SBATCH --partition=long

# For parallel jobs
##SBATCH --cpus-per-task=8
##SBATCH --nodes=2
##SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4


#SBATCH --output=hp_out.txt
#SBATCH --error=hp_err.txt

# For array jobs
# Array job containing 100 tasks, run max 10 tasks at the same time
#SBATCH --array=1-600

#### Your shell commands below this line ####
source nat_grad_env/bin_activate
python2.7 "File02_algorithm_Fig7.py"

