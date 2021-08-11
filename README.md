#  ML4CO - dual task

Code maintainer： Wentao Zhao (wz2543@columbia.edu)

This code combines the baseline model and Abhi's research code. Please do not distribute the code.
We apply a new graph convolutional neural network model for learning branch-and-bound variable selection policies.
Training is done through imitation learning and evolution strategy. 
The training process has three steps: sample generation, behavior cloning(bc), and evolution strategy(es).


### Instruction:
####  Clone this repository
```bash
git clone https://github.com/LoganZhao1997/ml4co_dual_task.git
cd ml4co_dual_task
```
Make sure instances are available on `instances`. You can download the instances [here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view).


#### Set-up your Python dependencies
```bash
source init.sh
```

#### Generate sample
`python bc/01_generate_dataset.py BENCHMARK`
Optional arguments:
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-j NJOBS`: number of parallel sample-generation jobs.


#### run behavior cloning
`python bc/02_train.py BENCHMARK`
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-g GPU`: CUDA GPU id (or -1 for CPU only)
When training, the file `bc/trained_models/$BENCHMARK/best_params.pkl` will be generated.


#### run evolution strategy
`python es/src/main.py BENCHMARK`


#### Evaluate
To evaluate the results, copy the trained models (`bc/trained_models/$BENCHMARK/best_params.pkl`) into the `agents` directory, which imitates the final submission format. 
Follow the evaluation pipeline instructions to evaluate the generated parameters.

