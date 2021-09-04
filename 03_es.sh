#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=es    # The job name.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --time=0:15:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.




# evolutionary strategy
module load singularity
source singularity/04evolutionary_strategy.sh $1