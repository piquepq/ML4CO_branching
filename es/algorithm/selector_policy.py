import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import ray

from es.algorithm.solution import Solution
from es.algorithm.distributions import Argmax
from env.mip import MIP
from env.episode import Episode
from env.brancher import Brancher
from env.selector import Selector
from env.meta import HOME


class SelectorPolicy(Solution):
    def __init__(self):
        self.model = nn.Sequential(
                        nn.Linear(88,64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32,1)
                    )

        # self.model = nn.Sequential(
        #         nn.Linear(88,128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64,32),
        #         nn.ReLU(),
        #         nn.Linear(32,1)
        #     )

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

    def act(self, state):
        pdparam = self.model(torch.from_numpy(state.astype(np.float32)))
        pdparam = pdparam.transpose(0,1)
        action_pd = Argmax(logits=pdparam)    
        action = action_pd.sample().item()
        # return np.random.randint(low=0, high=pdparam.shape[1])
        return action

    def save(self, path, verbose=True):
        if verbose:
            print("... saving models to %s" % (path))
        torch.save({
            'policy' :  self.model.state_dict()
        }, path)
        
    def load(self, path):
        # print("... loading models from %s" % (path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['policy'])

    def evaluate(self, instance):   
        from es.algorithm.brancher_policy import BrancherPolicy

        mip = MIP(instance=instance, seed=0)
        mip.configure_solver(visualization=False, presolving=False, separating=False, heuristics=False, conflict=False)
        
        brancherPolicy = BrancherPolicy()
        brancherPolicy.load(path=HOME+"/es/model/brancherIT.pth")
        brancher = Brancher(policy=brancherPolicy, verbose=False) 
        selector = Selector(policy=self, verbose=False)

        episode = Episode(mip, brancher, selector, verbose=False)
        score = episode.run()

        episodes = 1
        transitions = 1
        return score, episodes, transitions