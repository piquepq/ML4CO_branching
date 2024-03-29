import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import json
import ecole
from pathlib import Path

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from es.algorithm.solution import Solution
from common.environments import Branching as Environment
from common.rewards import TimeLimitDualIntegral as BoundIntegral
from es.config.config import EVAL_TIME_LIMT


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False



class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__('add')
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output


class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        output = self.output_module(variable_features).squeeze(-1)
        return output


class BrancherPolicy(Solution):
    def __init__(self):
        self.model = GNNPolicy()
        for param in self.model.parameters():
            param.requires_grad = False

        self.dim = sum(p.numel() for p in self.model.parameters())

    def size(self):
        return self.dim

    def get_params(self):
        params = parameters_to_vector(self.model.parameters()).numpy()
        assert params.dtype == np.float32
        return params

    def set_params(self, params):
        assert params.dtype == np.float32
        vector_to_parameters(torch.tensor(params), self.model.parameters())

    def save(self, path, verbose=True):
        if verbose:
            print("... saving models to %s" % (path))
        torch.save({
            'policy' :  self.model.state_dict()
        }, path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

        # # print("... loading models from %s" % (path))
        # checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['policy'])

    def evaluate(self, instance):
        observation_function = ecole.observation.NodeBipartite()
        integral_function = BoundIntegral()
        env = Environment(
            time_limit=EVAL_TIME_LIMT,
            observation_function=observation_function,
            reward_function=-integral_function,  # negated integral
        )

        # read the instance's initial primal and dual bounds from JSON file
        instance = Path(instance)
        with open(instance.with_name(instance.stem).with_suffix('.json')) as f:
            instance_info = json.load(f)

        # set up the reward function parameters for the instance
        initial_primal_bound = instance_info["primal_bound"]
        initial_dual_bound = instance_info["dual_bound"]
        objective_offset = 0

        integral_function.set_parameters(
                initial_primal_bound=initial_primal_bound,
                initial_dual_bound=initial_dual_bound,
                objective_offset=objective_offset)

        # reset the environment
        observation, action_set, reward, done, info = env.reset(str(instance), objective_limit=initial_primal_bound)
        cumulated_reward = 0  # discard initial reward

        # loop over the environment
        while not done:
            action = self.make_action(action_set, observation)
            observation, action_set, reward, done, info = env.step(action)
            cumulated_reward += reward

        # print(f"  cumulated reward (to be maximized): {cumulated_reward}")
        score = - cumulated_reward  # we want to minimize the score
        episodes = 1
        transitions = 1
        return score, episodes, transitions

    def make_action(self, action_set, observation):
        # mask variable features (no incumbent info)
        variable_features = observation.column_features
        variable_features = np.delete(variable_features, 14, axis=1)
        variable_features = np.delete(variable_features, 13, axis=1)

        constraint_features = torch.FloatTensor(observation.row_features)
        edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
        edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1))
        variable_features = torch.FloatTensor(variable_features)
        action_set = torch.LongTensor(np.array(action_set, dtype=np.int64))

        logits = self.model(constraint_features, edge_index, edge_attr, variable_features)
        logits = logits[action_set]
        action_idx = logits.argmax().item()
        action = action_set[action_idx]

        return action


