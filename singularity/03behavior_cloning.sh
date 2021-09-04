#!/bin/bash

display_usage() {
    echo -e "  BENCHMARK: problem benchmark to evaluate (item_placement, load_balancing, anonymous)"
    echo -e "  OPTIONS:"
    echo -e "    -s (--seed): random seed used to initialize the pseudo-random number generator"
    echo -e "    -g (--gpu): CUDA GPU id (or -1 for CPU only)"
}

# if less than two arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

COMMANDS="source /opt/mamba/init.bash; conda activate ml4co; python bc/02_train.py $@"
singularity exec --net singularity/base.sif bash -i -c "$COMMANDS"
