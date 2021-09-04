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
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

# set directory for instances
cd /rigel/seasdean/projects/ml4co/instances
export INSTANCES_PATH="$PWD"
cd ~/ml4co_dual_task

# es
COMMANDS="source /opt/mamba/init.bash; conda activate ml4co; python es/src/main.py $@"
singularity exec --bind $INSTANCES_PATH:/instances singularity/base.sif bash -i -c "$COMMANDS"
