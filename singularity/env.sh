#!/bin/sh

# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml

conda activate ml4co

# install pytorch with cuda 10.2
# conda install -y pytorch==1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -y pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# install pytorch geometric
conda install -y pytorch-geometric -c rusty1s -c conda-forge

# install ray
pip install "ray[default]"

# install package
pip install -e .


