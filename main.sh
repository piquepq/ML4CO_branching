#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=init_ml4co    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=20:00              # The time the job will take to run (here, 1 $
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.



# generate data samples
source singularity/02generate_data.sh item_placement -w 0

# behavior cloning
source singularity/03behavior_cloning.sh item_placement -g -1

# evolutionary strategy
source singularity/04evolutionary_strategy.sh item_placement