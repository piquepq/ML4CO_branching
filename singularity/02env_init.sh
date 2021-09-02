#!/bin/bash


export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1

singularity exec singularity/base.sif bash -i -c "source /opt/mamba/init.bash; source init.sh"
