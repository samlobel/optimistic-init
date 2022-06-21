import numpy as np

def uniform_scaling(*args, **kwargs):
    return 1.


def action_scaling(env, action_scaler):
    """
    This is actually going to just be "action scaling". Because,
    it's all about the ratio, and the ratio doesn't change!
    """
    try:
        state_dim = len(env.observation_space.low)
    except AttributeError:
        print("Using dm_control so need to get state_dim differently")
        state_dim = len(env.observation_space['observations'].low)

    action_dim = len(env.action_space.low)

    action_scaler = float(action_scaler)

    state_scaler_array = np.ones((state_dim,), dtype=np.float32)
    action_scaler_array = np.ones((action_dim,), dtype=np.float32) * action_scaler

    return np.concatenate([state_scaler_array, action_scaler_array], axis=0)

def per_dim_scaling(env, *args):
    try:
        state_dim = len(env.observation_space.low)
    except AttributeError:
        print("Using dm_control so need to get state_dim differently")
        state_dim = len(env.observation_space['observations'].low)
    action_dim = len(env.action_space.low)
    assert len(args) == state_dim + action_dim
    return np.array(args, dtype=np.float32)


"""
This has an interesting interface -- scaling_string is a string where the arguments are double-underscore-separated.
That lets us pass stuff in through a CLI interface a bit easier.
"""

_SCALING_FUNCTIONS = {
    'action_scaling': action_scaling,
    'per_dim_scaling': per_dim_scaling,
}

def get_scaling_array(env, scaling_function_string):
    scaling_string_parsed = scaling_function_string.split("__")
    scaling_method, scaling_args = scaling_string_parsed[0], scaling_string_parsed[1:]
    scaling_array = _SCALING_FUNCTIONS[scaling_method](env, *scaling_args)
    return scaling_array
