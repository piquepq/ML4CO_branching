from torch import distributions
import torch

class Argmax(distributions.Categorical):
    '''
    Special distribution class for argmax sampling, where probability is always 1 for the argmax.
    NOTE although argmax is not a sampling distribution, this implementation is for API consistency.
    '''

    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            new_probs = torch.zeros_like(probs, dtype=torch.float)
            new_probs[probs == probs.max(dim=-1, keepdim=True)[0]] = 1.0
            probs = new_probs
        elif logits is not None:
            new_logits = torch.full_like(logits, -1e8, dtype=torch.float)
            new_logits[logits == logits.max(dim=-1, keepdim=True)[0]] = 1.0
            logits = new_logits

        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

