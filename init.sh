#!/bin/sh
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=seasdean	# The account name for the job.
#SBATCH --job-name=init_ml4co    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=20:00              # The time the job will take to run (here, 1 $
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.

# module load
module load anaconda/3-2019.03
conda init bash


# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml

conda activate ml4co

# install pytorch with cuda 10.2
conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch

# install pytorch geometric
conda install pytorch-geometric -c rusty1s -c conda-forge

# install ray
pip install "ray[default]"

# install package
pip install -e .


