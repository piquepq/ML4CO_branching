#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=init_ml4co    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=20:00              # The time the job will take to run (here, 1 $
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.


## unzip instances
cd /rigel/seasdean/projects/ml4co
tar -xzf instances.tar.gz
cd ~/ml4co_dual_task