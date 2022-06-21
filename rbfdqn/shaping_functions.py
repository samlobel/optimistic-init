"""
These will be functions maybe for each class that specify a tighter upper bound for q_max,
as opposed to q_max everywhere. These will just be state-based... Easy example would using
distance-from-goal if you know the maximum step size. The benefit over a dense reward is
that this will be "provably" convergent even if you'd think there were local optima.

Note that these should probably depend on gamma for the most part. Don't know how to
add that easily.
"""

import numpy as np
import torch

def mcar_shaping_1(states, gamma=0.99):
    """
    Max speed = 0.07, and positoin range is 1.8.
    That means that if we're at position (0.6 - d),
    the number of steps we need to take is (0.6 - d) / 0.07, which at most is
    1.8/0.07 = 25 steps away, meaning that it will value the left side at 0.77
    and the right side at 1 to start.
    """
    x_coord_states = states[:,0]

    min_steps_away = (0.45 - x_coord_states) / 0.07
    min_steps_away = torch.clamp(min_steps_away, min=0.)

    Q_max = 100. * (gamma ** min_steps_away)
    return Q_max

def mcar_shaping_bad(states, gamma=0.99):
    """
    The opposite of our good shaping function. It will encourage you to spend time on the left side of the screen.
    How? To preserve the property, we'll set the max Q to be 100 at the source, and more elsewhere.
    Just reverse the steps away part. Nice.
    """
    x_coord_states = states[:,0]

    # Should add this back in at some point
    min_steps_away = (0.45 - x_coord_states) / 0.07
    min_steps_away = torch.clamp(min_steps_away, min=0.)
    reversed_min_steps_away = -1 * min_steps_away

    Q_max = 100. * (gamma ** reversed_min_steps_away)
    return Q_max



class ShapingFunctions:

    """Okay, I want this to take in a tensor. Much faster that way. """

    shaping_functions = {
        'MountainCarContinuous-v0': {
            'default': mcar_shaping_1,
            'max_steps_away': mcar_shaping_1,
            'reversed_max_steps_away': mcar_shaping_bad,
        }
    }

    def __init__(self, env_name, gamma, func_name=None):

        assert env_name in self.shaping_functions.keys(), env_name

        self.env_name = env_name
        self.gamma = gamma
        self.func_name = func_name

        self._assign_shaping_function()
    
    def _assign_shaping_function(self):
        sfd = self.shaping_functions[self.env_name]

        func_name = 'default' if self.func_name is None else self.func_name
        sf = sfd[func_name]
        self._shaping_function = sf

    def get_values(self, states):
        reshaped = False
        if len(states.shape) == 1:
            reshaped = True
            states = states.view(1,-1)

        shaping_values = self._shaping_function(states, self.gamma)

        if reshaped == True:
            shaping_values = shaping_values.view(-1)
        
        return shaping_values

