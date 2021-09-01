import os
import sys
import ray
import glob
import wandb
import argparse

from bc.utilities import log
from es.model.brancher_policy import BrancherPolicy as Policy
from es.algorithm.trainer import Trainer
from es.config.config import NUM_WORKER, STEP_SIZE, EPOCHS, MIN_EVAL, NOISE_STD, SEED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    args = parser.parse_args()

    # best parameters path
    if args.problem == 'item_placement':
        policy_path = 'bc/trained_models/item_placement/best_params.pkl'
        instances_path = 'instances/1_item_placement/valid/*.mps.gz'

    elif args.problem == 'load_balancing':
        policy_path = 'bc/trained_models/load_balancing/best_params.pkl'
        instances_path = 'instances/2_load_balancing/valid/*.mps.gz'

    elif args.problem == 'anonymous':
        policy_path = 'bc/trained_models/anonymous/best_params.pkl'
        instances_path = 'instances/3_anonymous/valid/*.mps.gz'

    else:
        raise NotImplementedError

    # set content root path
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    policy_path = os.path.join(DIR, policy_path)
    instances_path = os.path.join(DIR, instances_path)
    instances_valid = glob.glob(instances_path)

    # set up log
    run_dir = os.path.dirname(policy_path)
    logfile = os.path.join(run_dir, 'es_train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {EPOCHS}", logfile)
    log(f"num_workers: {NUM_WORKER}", logfile)
    log(f"step_size: {STEP_SIZE}", logfile)
    log(f"min_evaluations: {MIN_EVAL}", logfile)
    log(f"noise_std: {NOISE_STD}", logfile)
    log(f"seed {SEED}", logfile)

    # initialize wandb
    wandb.init(project='ml4co-dual-es', entity='ml4co')

    ray.init()
    trainer = Trainer(Policy=Policy, policy_path=policy_path, instances=instances_valid, seed=SEED, num_workers=NUM_WORKER,
                      step_size=STEP_SIZE, count=10000000,
                      min_task_runtime=100000, logfile=logfile)
    trainer.train(epochs=EPOCHS, min_evaluations=MIN_EVAL, noise_std=NOISE_STD)
