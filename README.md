#  ML4CO - dual task

Code maintainer： Wentao Zhao (wz2543@columbia.edu)

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
source 00_init.sh
```


## Set up the env on Habanero
####  Transfer the file
You need to transfer instances.tar.gz from your local machine to Habanero. 
Find the instruction [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+-+Working+on+Habanero#HabaneroWorkingonHabanero-TransferringFiles).
ATTENTION: you account only have 10 GB storage, so you need to put it in the seasdean shared file.
Use the command below to transfer the file:
```bash
scp instances.tar.gz <UNI>@habaxfer.rcs.columbia.edu:/rigel/seasdean/projects/ml4co
```
This command automatically creates a folder called ml4co in /rigel/seasdean/projects/ with instances.tar.gz. 
(Log in your account AFTER transferring)

####  Clone this repository in your home directory on Habanero
```bash
cd ~
git clone https://github.com/LoganZhao1997/ml4co_dual_task.git
```

####  Initialize the env
```bash
cd ~/ml4co_dual_task
sbatch 00_init.sh
```
If this command is successful you will see a message that says: "Submitted batch job 'Job ID' ".

Use the command below to check whether the initialization succeeds:
```bash
scontrol show job <Job ID>
```

## Instruction for running the code
ATTENTION: revise the 01_generate.sh, 02_bc.sh, 03_es.sh to decide
how many CPU, GPU, memory you need and the running time.
See instruction [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+-+Submitting+Jobs).

#### Generate sample
```bash
sbatch 01_generate.sh BENCHMARK
```
To determine if the job is completed you can run:
```bash
scontrol show job <Job ID>
```
Check "JobState = " to determine if it's still RUNNING or COMPLETED. 
  
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

#### Visualize the training 
You can zip the wandb records and download it to your local machine
```bash
cd ~/ml4co_dual_task
zip -r wandb.zip wandb
```
Open another terminal without connecting with habanero and use scp to download files:
```bash
scp <UNI>@habanero.rcs.columbia.edu:~/ml4co_dual_task/wandb.zip <destination>
```

Unzip the wandb.zip and use wandb sync to upload the training performance
```bash
wandb sync [PATH]
```
#### Evaluate
Follow the evaluation pipeline instructions to evaluate the generated parameters.

