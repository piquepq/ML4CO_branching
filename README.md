#  ML4CO - dual task

Code maintainer： Wentao Zhao (wz2543@columbia.edu)

This code combines the baseline model and Abhi's research code. Please do not distribute the code.
We apply a new graph convolutional neural network model for learning branch-and-bound variable selection policies.
Training is done through imitation learning and evolution strategy. 
The training process has three steps: sample generation, behavior cloning(bc), and evolution strategy(es).


## Set up the env on local machine
####  Clone this repository
```bash
git clone https://github.com/LoganZhao1997/ml4co_dual_task.git
cd ml4co_dual_task
```

Download the training instances [here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view), and extract the archive at the root of this repo
```bash
tar -xzf instances.tar.gz
```

#### Set-up your Python dependencies
```bash
source init.sh
```


## Set up the env on Habanero
####  Transfer the file
You need to transfer instances.tar.gz from your local machine to Habanero. 
Find the instruction [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+-+Working+on+Habanero#HabaneroWorkingonHabanero-TransferringFiles).
ATTENTION: you account only have 10 GB storage, so you need to put it in the seasdean shared file.
Use the command below to transfer the file:
```bash
scp MyDataFile <UNI>@habaxfer.rcs.columbia.edu:/seasdean/projects/ml4co
```
(Log in your account AFTER transferring)

####  Clone this repository
```bash
git clone https://github.com/LoganZhao1997/ml4co_dual_task.git
```

####  Initialize the env
```bash
cd ~/ml4co_dual_task
sbatch init.sh
```
Use the command below to check whether the initialization succeeds:
```bash
scontrol show job [job ID]
```

## Instruction for running the code
ATTENTION: revise the 01_generate.sh, 02_bc.sh, 03_es.sh to decide
how many CPU, GPU, memory you need and the running time.
See instruction [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+-+Submitting+Jobs).

#### Generate sample
```bash
sbatch 01_generate.sh BENCHMARK
```

#### Wandb login
Log in your wandb account before training.
```bash
wandb login
```

#### Behavior cloning
```bash
sbatch 02_bc.sh BENCHMARK
```
When the job is completed, use the command below to sync the training performance.
You can find YOUR_RUN_DIRECTORY at the last line of slurm output.
```bash
wandb sync YOUR_RUN_DIRECTORY
```

#### Evolution strategy
```bash
sbatch 03_es.sh BENCHMARK
```
When the job is completed, sync the training performance.
```bash
wandb sync YOUR_RUN_DIRECTORY
```

#### Evaluate
Follow the evaluation pipeline instructions to evaluate the generated parameters.

