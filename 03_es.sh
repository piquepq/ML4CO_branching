#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=es    # The job name.
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -c 24                     # The number of cpu cores to use.
#SBATCH --time=0:0:30              # The time the job will take to run.
#SBATCH --mem = 128gb         # The memory the job will use per node.




# evolutionary strategy
module load singularity
source singularity/04evolutionary_strategy.sh $1