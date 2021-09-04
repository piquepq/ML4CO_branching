#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=init_ml4co    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=30:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb        # The memory the job will use per cpu core.


# transfer instances to seasdean shared file and unzip it
 cd /rigel/seasdean/projects/ml4co
 tar -xzf instances.tar.gz

# install wandb
module load anaconda
conda install wandb

# initialize the environment
cd ~/ml4co_dual_task
module load singularity
source singularity/01env_init.sh

