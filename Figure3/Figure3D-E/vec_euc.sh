#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=kreutzer@pyl.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end

# Job name
#SBATCH --job-name="vectorplot EG"

# Runtime and memory
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G

# Partition
##SBATCH --partition=long

# For parallel jobs
##SBATCH --cpus-per-task=8
##SBATCH --nodes=2
##SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4


#SBATCH --output=vec_euc_out.txt
#SBATCH --error=vec_euc_error.txt

# For array jobs
# Array job containing 100 tasks, run max 10 tasks at the same time
#SBATCH --array=1-2000

#### Your shell commands below this line ####

python2.7 "File03_vectorplot_euc.py"

