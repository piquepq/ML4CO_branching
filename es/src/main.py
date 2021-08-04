import os

import ray
import argparse
import glob
import sys

from es.algorithm.trainer import Trainer


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'problem',
    #     help='MILP instance type to process.',
    #     choices=['item_placement', 'load_balancing', 'anonymous'],
    # )
    # args = parser.parse_args()
    #
    # # best parameters path
    # if args.problem == 'item_placement':
    #     policy_path = '../../bc/trained_models/item_placement/best_params.pkl'
    #     instances_valid = glob.glob('../instances/1_item_placement/valid/*.mps.gz')
    #
    # elif args.problem == 'load_balancing':
    #     policy_path = '../../bc/trained_models/load_balancing/best_params.pkl'
    #     instances_valid = glob.glob('../instances/2_load_balancing/valid/*.mps.gz')
    #
    # elif args.problem == 'anonymous':
    #     policy_path = '../../bc/trained_models/anonymous/best_params.pkl'
    #     instances_valid = glob.glob('../instances/3_anonymous/valid/*.mps.gz')
    #
    # else:
    #     raise NotImplementedError

    ray.init()

    DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    policy_path = os.path.join(DIR, 'bc/trained_models/item_placement/best_params.pkl')
    instance_path = os.path.join(DIR, 'instances/1_item_placement/valid/*.mps.gz')
    instances_valid = glob.glob(instance_path)

    from es.model.brancher_policy import BrancherPolicy as Policy

    trainer = Trainer(Policy=Policy, policy_path=policy_path, instances=instances_valid, seed=1, num_workers=3,
                      step_size=0.1, count=10000000,
                      min_task_runtime=100000)
    trainer.train(epochs=1, min_evaluations=10, noise_std=0.1)
