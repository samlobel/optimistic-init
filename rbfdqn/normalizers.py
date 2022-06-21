import numpy as np
import torch

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    We're switching this to be torch because that's what we use it on.
    """
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        self.mean = torch.zeros(*shape, device=device)
        self.var = torch.ones(*shape, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
    
    def normalize_batch(self, x):
        """
        Normalizes with mean and standard deviation
        """
        normalized_x = x - self.mean.view(1, -1)
        normalized_x = normalized_x / (torch.sqrt(self.var.view(1, -1) + 1e-16))
        return normalized_x
        

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count