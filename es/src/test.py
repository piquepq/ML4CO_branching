import glob
import os
import json
import numpy as np
from pathlib import Path
from es.model.brancher_policy import BrancherPolicy as Policy

DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
policy_path = os.path.join(DIR, 'bc/trained_models/item_placement/best_params.pkl')
instance_path = os.path.join(DIR, 'instances/1_item_placement/valid/*.mps.gz')
instances_valid = glob.glob(instance_path)
# policy_path = '../../bc/trained_models/item_placement/best_params.pkl'
# instances_valid = glob.glob('../instances/1_item_placement/valid/*.mps.gz')

solution = Policy()
instance = instances_valid[0]

# rng = np.random.RandomState(0)
# instance = Path(rng.choice(instances_valid))
# with open(instance.with_name(instance.stem).with_suffix('.json')) as f:
#     instance_info = json.load(f)

# solution.load(path=policy_path)
#
# theta = solution.get_params()
# print(theta)
# solution.set_params(theta)

print(solution.size())

