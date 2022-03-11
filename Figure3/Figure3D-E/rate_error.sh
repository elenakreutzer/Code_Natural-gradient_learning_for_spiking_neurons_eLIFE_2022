#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=kreutzer@pyl.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end

# Job name
#SBATCH --job-name="LC rate error"

# Runtime and memory
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G

# Partition


#SBATCH --output=rate_out.txt
#SBATCH --error=rate_error.txt

# For array jobs
# Array job containing 100 tasks, run max 10 tasks at the same time
#SBATCH --array=1-500

#### Your shell commands below this line ####

python2.7 "File05_main_simulation.py"
