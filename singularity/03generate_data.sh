#!/bin/bash


export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_BIND="${SINGULARITY_BIND},$(realpath instances):$SINGULARITY_HOME/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

COMMANDS="source /opt/mamba/init.bash; conda activate ml4co; python bc/01_generate_dataset.py item_placement"
singularity exec --net singularity/base.sif bash -i -c "$COMMANDS"
