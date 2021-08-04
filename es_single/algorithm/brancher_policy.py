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


class EmbeddingModule(nn.Module):
    """
    This class will be used for both variable and constraint embedding
    """
    def __init__(self, n_feats, emb_size, device):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)

class EdgeEmbeddingModule(nn.Module):
    """
    This class will only be used for edge embedding
    """
    def __init__(self, n_feats, device):
        super().__init__()
        self.pre_norm_layer = nn.BatchNorm1d(n_feats)

    def forward(self, input):
        # return input
        return self.pre_norm_layer(input)

class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, right_to_left=False, device=None):
        super().__init__()
        self.iterations = 0
        self.device = device
        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.feature_module_left.weight)

        self.feature_module_edge = nn.Linear(1, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_edge.weight)

        self.feature_module_right = nn.Linear(self.emb_size, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_right.weight)

        self.feature_model_final = nn.Sequential(
            nn.BatchNorm1d(self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )

        self.post_conv_module = nn.BatchNorm1d(self.emb_size)

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU()
        )

    def forward(self, left_features, edge_indices, edge_features, right_features, output_size):
        """
        Performs a partial graph convolution on the given bipartite graph.

        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)

        """
        self.iterations += 1
        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        # constructing messages
        # step 1. build messages from edge features (every edge has a message)
        joint_features = self.feature_module_edge(edge_features)
        # step 2. add variable features to the message (only add those variable features to this message if the edge is connected to this variable)
        joint_features.add_(self.feature_module_right(right_features)[edge_indices[1]])
        # step 3. add constraint features to the message (only add those constraint features to this message if the edge is connected to this constraint)
        joint_features.add_(self.feature_module_left(left_features)[edge_indices[0]])
        
        # apply layer on these messages
        joint_features = self.feature_model_final(joint_features)

        # aggregate the messages into the constraint / variable nodes
        conv_output = torch.zeros([output_size, self.emb_size]).to(self.device).index_add(0, edge_indices[scatter_dim], joint_features)
        conv_output = self.post_conv_module(conv_output)
        output = torch.cat((conv_output, prev_features), dim=1)
        return self.output_module(output)

class GraphNeuralNetwork(nn.Module):
    def __init__(self, device):
        super(GraphNeuralNetwork, self).__init__()
        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19

        self.cons_embedding = EmbeddingModule(self.cons_nfeats, self.emb_size, device).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats, device).to(device)
        self.var_embedding = EmbeddingModule(self.var_nfeats, self.emb_size, device).to(device)

        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, right_to_left=True, device=device)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, device=device)

        self.var_output_size = 1
        self.output = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.var_output_size)
        ).to(device)

    def forward(self, batch):
        state, candidates, constraints_per_sample, variables_per_sample, candidates_per_sample = batch["state"], batch["candidates"], batch["constraints/sample"], batch["variables/sample"], batch["candidates/sample"]
        constraint_features, edge_indices, edge_features, variable_features = state

        total_constraints = torch.sum(constraints_per_sample)
        total_variables = torch.sum(variables_per_sample)

        # embeddings
        constraint_features = self.cons_embedding(constraint_features.float())
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features.float())

        # convolution
        constraint_features = self.conv_v_to_c(constraint_features, edge_indices, edge_features, variable_features, output_size=total_constraints)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features, output_size=total_variables)

        # variable latent features to logits
        variable_logits = self.output(variable_features)
        # only grab logits for the variables that are candidates for branching
        candidate_logits = variable_logits.squeeze(1)[candidates.long()].unsqueeze(1)
        # reshape into (N samples, D) where D = max(number of branching candidates) over all samples
        candidate_logits = self.pad_output(candidate_logits, candidates_per_sample)

        return candidate_logits
        
    def pad_output(self, logits, candidates_per_sample, pad_value=-1e8):
        n_cands_max = torch.max(candidates_per_sample).item()
        logits = logits.reshape(1, -1)
        logits = torch.split(logits, candidates_per_sample.squeeze().tolist(), dim=1)
        output = [0]*candidates_per_sample.shape[0]
        for i in range(len(output)):
            output[i] = torch.nn.functional.pad(logits[i],(0, n_cands_max - logits[i].shape[1]),'constant', pad_value)
        output = torch.cat(output, dim=0)
        return output

    def save(self, path):
        # print("... saving models to %s" % (path))
        torch.save({
            'policy' :  self.state_dict()
        }, path)
            
    def load(self, path):
        print("... loading models from %s" % (path))
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['policy'])

    @staticmethod
    def to_torch_batch(batch, device, episodic=False):
        if not episodic:
            return {
                "states"        :   GraphNeuralNetwork.process_state(batch["states"], device),
                "actions"       :   GraphNeuralNetwork.process_action(batch["actions"], device),
                "rewards"       :   GraphNeuralNetwork.process_reward(batch["rewards"], device),
                "next_states"   :   GraphNeuralNetwork.process_state(batch["next_states"], device),
                "dones"         :   GraphNeuralNetwork.process_done(batch["dones"], device)
            }
        else:
            raise Exception("episodic to_torch_batch not implemented")
    
    @staticmethod
    def process(array: list, device):
        return torch.from_numpy(np.array(array).astype(np.float32)).to(device)

    @staticmethod
    def process_state(states, device):
        c_features = []
        e_indices = []
        e_features = []
        v_features = []
        candidates = []

        n_cs_per_sample = []    # need this to shift edge indices accordingly for constraints
        n_vs_per_sample = []    # need this to shift edge indices accordingly for variables
        n_cands_per_sample = []

        scores = []

        for state in states:
            c_f, e_i, e_f, v_f, cands = state

            c_f = np.nan_to_num(c_f)
            c_f = c_f.astype('float32')

            e_i = np.nan_to_num(e_i)
            e_i = e_i.astype('int32')

            e_f = np.nan_to_num(e_f)
            e_f = e_f.astype('float32')

            v_f = np.nan_to_num(v_f)
            v_f = v_f.astype('float32')

            cands = cands.astype('int32')

            n_cs = np.asarray(c_f.shape[0])
            n_vs = np.asarray(v_f.shape[0])
            n_cands = np.asarray(cands.shape[0])

            c_features.append(c_f)
            e_indices.append(e_i)
            e_features.append(e_f)
            v_features.append(v_f)
            candidates.append(cands)
            n_cs_per_sample.append(n_cs)
            n_vs_per_sample.append(n_vs)
            n_cands_per_sample.append(n_cands)

        # concatenate samples in one big graph
        c_features = np.concatenate(c_features, axis=0)
        v_features = np.concatenate(v_features, axis=0)
        e_features = np.concatenate(e_features, axis=0)
       
        # edge indices have to be adjusted accordingly
        cv_shift = np.cumsum([
                [0] + n_cs_per_sample[:-1],
                [0] + n_vs_per_sample[:-1]
            ], axis=1)
        e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
            for j, e_ind in enumerate(e_indices)], axis=1)
        e_indices = e_indices.astype(np.int32)

        # candidate indices as well
        candidates = np.concatenate([cands + shift
            for cands, shift in zip(candidates, cv_shift[1])])
        
        # metadata
        n_cs_per_sample = np.array(n_cs_per_sample)[..., np.newaxis]
        n_vs_per_sample = np.array(n_vs_per_sample)[..., np.newaxis]
        n_cands_per_sample = np.array(n_cands_per_sample)[..., np.newaxis]

        batch = {
            "state"                :   (c_features, e_indices, e_features, v_features),
            "candidates"            :   candidates,
            "constraints/sample"    :   n_cs_per_sample,
            "variables/sample"      :   n_vs_per_sample,
            "candidates/sample"     :   n_cands_per_sample
        }

        return GraphNeuralNetwork.toTensor(batch, device)
        
    @staticmethod
    def toTensor(batch, device):
        c_f, e_i, e_f, v_f = batch["state"]
        return {
            "state"            :   (torch.from_numpy(c_f).to(device), torch.from_numpy(e_i).to(device).long(), torch.from_numpy(e_f).to(device), torch.from_numpy(v_f).to(device)),
            "candidates"        :   torch.from_numpy(batch["candidates"]).to(device),
            "constraints/sample":   torch.from_numpy(batch["constraints/sample"]).to(device),
            "variables/sample"  :   torch.from_numpy(batch["variables/sample"]).to(device),
            "candidates/sample" :   torch.from_numpy(batch["candidates/sample"]).to(device)
        }

    @staticmethod
    def process_action(actions, device):
        return GraphNeuralNetwork.process(actions, device)
    
    @staticmethod
    def process_reward(rewards, device):
        return GraphNeuralNetwork.process(rewards, device)
    
    @staticmethod
    def process_done(dones, device):
        return GraphNeuralNetwork.process(dones, device)


class BrancherPolicy(Solution):
    def __init__(self):
        self.model = GraphNeuralNetwork(device=torch.device("cpu"))
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
        states = [state, state]
        states = GraphNeuralNetwork.process_state(states, torch.device("cpu"))
        pdparam = self.model(states)
        action_pd = Argmax(logits=pdparam)    
        action = action_pd.sample().cpu().squeeze().numpy()
        return action[0]

    def save(self, path, verbose=True):
        if verbose:
            print("... saving models to %s" % (path))
        torch.save({
            'policy' :  self.model.state_dict()
        }, path)
        
    def load(self, path):
        # print("... loading models from %s" % (path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['policy'])
 
    def evaluate(self, instance):   
        from es.algorithm.selector_policy import SelectorPolicy

        mip = MIP(instance=instance, seed=0)
        mip.configure_solver(visualization=False, presolving=False, separating=False, heuristics=False, conflict=False)
        
        brancher = Brancher(policy=self, verbose=False) 
        selectorPolicy = SelectorPolicy()
        selectorPolicy.load(path=HOME+"/es/model/selectorIT.pth")
        selector = Selector(policy=selectorPolicy, verbose=False)

        episode = Episode(mip, brancher, selector, verbose=False)
        score = episode.run()

        episodes = 1
        transitions = 1
        return score, episodes, transitions
