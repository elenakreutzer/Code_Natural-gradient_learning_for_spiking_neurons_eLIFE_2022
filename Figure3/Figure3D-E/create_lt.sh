#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=kreutzer@pyl.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end

# Job name
#SBATCH --job-name="Learning Curves for two Dendrites"

# Runtime and memory
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2G

# Partition
##SBATCH --partition=long

# For parallel jobs
##SBATCH --cpus-per-task=8
##SBATCH --nodes=2
##SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4


##SBATCH --output=outputfiles/dendritesout.txt
##SBATCH --error=errorfiles/dendriteserror.txt

# For array jobs
## Array job containing 100 tasks, run max 10 tasks at the same time
##SBATCH --array=7,39,78,390,780,3900,7800
##SBATCH --array=5,26,52,260,520,2600
#SBATCH
#### Your shell commands below this line ####

source nat_grad_env7bin/activate
python2.7 "File02_lt_vectorplot.py"

