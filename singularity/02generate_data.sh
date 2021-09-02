#!/bin/bash


display_usage() {
    echo -e "  BENCHMARK: problem benchmark to evaluate (item_placement, load_balancing, anonymous)"
    echo -e "    -w (--where): where you run the code, 0 for local machine, 1 for HPC"
    echo -e "  OPTIONS:"
    echo -e "    -s (--seed): random seed used to initialize the pseudo-random number generator"
    echo -e "    -j (--njob): number of parallel sample-generation jobs"

}

# if less than two arguments supplied, display usage
if [  $# -lt 2 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
# export SINGULARITY_BIND="${SINGULARITY_BIND},$(realpath instances):$SINGULARITY_HOME/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

COMMANDS="source /opt/mamba/init.bash; conda activate ml4co; python bc/01_generate_dataset.py $@"
singularity exec --net singularity/base.sif bash -i -c "$COMMANDS"
