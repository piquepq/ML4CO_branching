#!/bin/bash

display_usage() {
    echo -e "  BENCHMARK: problem benchmark to evaluate (item_placement, load_balancing, anonymous)"
}

# if less than two arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath $PWD`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
# export SINGULARITY_BIND="${SINGULARITY_BIND},$PWD/../../../seasdean/projects/ml4co/instances:$SINGULARITY_HOME/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

COMMANDS="source /opt/mamba/init.bash; conda activate ml4co; python es/src/main.py $@"
singularity exec --net singularity/base.sif bash -i -c "$COMMANDS"
