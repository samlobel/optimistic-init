from collections import defaultdict
import numpy
import os
import sys
import torch

from rbfdqn.utils import boolify

def action_checker(env):
    """
    I've changed it so that it just needs to be symmetric, and it does per-feature action scaling.
    """
    print("action range:",env.action_space.low,env.action_space.high)
    for l,h in zip(env.action_space.low,env.action_space.high):
        if l!=-h:
            print("asymetric action space")
            print("don't know how to deal with it")
            assert False


def get_hyper_parameters(name,alg):
    meta_params={}
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(THIS_DIR, alg + "_hyper_parameters",
                            name + ".hyper")
    with open(filepath) as f:
        lines = [line.rstrip('\n') for line in f]
        for l in lines:
            parameter_name,parameter_value,parameter_type=(l.split(','))
            if parameter_type=='string':
                meta_params[parameter_name]=str(parameter_value)
            elif parameter_type=='integer':
                meta_params[parameter_name]=int(parameter_value)
            elif parameter_type=='float':
                meta_params[parameter_name]=float(parameter_value)
            elif parameter_type=='boolean':
                meta_params[parameter_name]=boolify(parameter_value)
            else:
                print("unknown parameter type ... aborting")
                print(l)
                sys.exit(1)
    return meta_params

def save_hyper_parameters(params, seed):
    hyperparams_filename = '{}__seed_{}.hyper'.format(
        params['hyper_parameters_name'],
        seed,
    )
    hyperparams_path = os.path.join(params['hyperparams_dir'], hyperparams_filename)
    with open(hyperparams_path, 'w') as file:
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                int: 'integer',
                str: 'string',
                float: 'float',
                bool: 'boolean',
            })[type(value)] # yapf: disable
            if type_str is not None:
                file.write("{},{},{}\n".format(name, value, type_str))

def sync_networks(target, online, alpha, copy = False):
    if copy == True:
        for online_param, target_param in zip(online.parameters(),
                                              target.parameters()):
            target_param.data.copy_(online_param.data)
    elif copy == False:
        for online_param, target_param in zip(online.parameters(),
                                              target.parameters()):
            target_param.data.copy_(alpha * online_param.data +
                                    (1 - alpha) * target_param.data)

def set_random_seed(meta_params):
    seed_number=meta_params['seed_number']
    import numpy
    numpy.random.seed(seed_number)
    import random
    random.seed(seed_number)
    import torch
    torch.manual_seed(seed_number)
    meta_params['env'].seed(seed_number)

class Reshape(torch.nn.Module):
	"""
	Description:
		Module that returns a view of the input which has a different size    Parameters:
		- args : Int...
			The desired size
	"""
	def __init__(self, *args):
		super().__init__()
		self.shape = args

	def __repr__(self):
		s = self.__class__.__name__
		s += '{}'.format(self.shape)
		return s

	def forward(self, x):
		return x.view(*self.shape)
