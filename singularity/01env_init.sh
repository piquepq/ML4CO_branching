#!/bin/bash

# pull the prebuilt image
cd singularity
singularity pull library://columbia_combinators/default/columbia_combinators:latest
mv columbia_combinators_latest.sif base.sif
cd ..

# set environment variables
export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1

# initialize the environment
singularity exec singularity/base.sif bash -i -c "source /opt/mamba/init.bash; source singularity/env.sh"
