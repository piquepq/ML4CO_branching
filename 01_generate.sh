#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=generate_sample    # The job name.
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -c 24                     # The number of cpu cores to use.
#SBATCH --time=0:0:30              # The time the job will take to run.
#SBATCH --mem = 128gb         # The memory the job will use per node.



# generate data samples
module load singularity
source singularity/02generate_data.sh $1 -j 24
# `-j NJOBS`: number of parallel sample-generation jobs.

