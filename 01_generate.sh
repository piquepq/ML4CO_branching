#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=generate_sample    # The job name.
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -c 24                     # The number of cpu cores to use.
#SBATCH --time=0:30:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.



# generate data samples
module load singularity
source singularity/02generate_data.sh $1

