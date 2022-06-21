import os
import pickle
from collections import OrderedDict
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import gym
import dm2gym


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("String '{}' is not a known bool value.".format(s))


def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


def update_param(params, arg_name, arg_value):
    if arg_name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".
            format(arg_name))
    else:
        print("Updating parameter '{}' to {}".format(arg_name, arg_value))
    converted_arg_value = autoconvert(arg_value)
    if type(params[arg_name]) != type(converted_arg_value):
        error_str = f"Old and new type must match! Got {type(converted_arg_value)}, expected {type(params[arg_name])}, for {arg_name}"
        raise ValueError(error_str)
    params[arg_name] = converted_arg_value


class DMSettlingWrapper(gym.Wrapper):
    """
    Allows for many "no-op" actions before the actual episode begins. Complicated by the fact that
    dm_control does internal "resets", and that we want to preserve the wrapping time-limit.
    """
    def __init__(self, env, random_steps=5000, reset_on_reward=False):
        assert isinstance(env, gym.wrappers.TimeLimit)
        assert isinstance(env.env, dm2gym.envs.dm_suite_env.DMSuiteEnv)
        super(DMSettlingWrapper, self).__init__(env)
        print(f"old step limit: {env.env.env._step_limit}")
        env.env.env._step_limit = float("inf")
        print(f"new step limit: {env.env.env._step_limit}")
        print(f"what is self env? {self.env}")

        self._random_steps = random_steps
        self.reset_on_reward = reset_on_reward
        self.MAX_RESETS = 10

    @staticmethod
    def sanitize_kwargs(kwargs):
        kwargs_copy = dict(kwargs)
        try:
            kwargs_copy.pop("num_resets")
        except KeyError:
            pass
        return kwargs_copy

    def unwrap_env(self):
        env = self.env
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        assert isinstance(env, dm2gym.envs.dm_suite_env.DMSuiteEnv)
        return env

    def reset(self, *args, **kwargs):
        # num_resets breaks the super's reset
        kwargs_copy = DMSettlingWrapper.sanitize_kwargs(kwargs)
        state = super(DMSettlingWrapper, self).reset(*args, **kwargs_copy)
        env = self.unwrap_env()
        noop_action = np.zeros_like(np.array(env.action_space.sample()))
        reward = 0
        for _ in range(self._random_steps):
            # print('doing noop thing')
            state, reward, done, info = env.step(noop_action)
            if done:
                print("bad!")
                raise Exception("you did something wrong!")

        if reward != 0:
            print("REWARD IS NOT ZERO ")
            num_resets = kwargs.get("num_resets", 0)
            if num_resets < self.MAX_RESETS:
                kwargs["num_resets"] = num_resets + 1
                print(f"Resetting # {kwargs['num_resets']}")
                return self.reset(*args, **kwargs)
            else:
                print(
                    f"But continuing because {num_resets}/{self.MAX_RESETS} done already"
                )
        assert self.env._elapsed_steps == 0
        return state


class DMSuiteUnwrapper(gym.Wrapper):
    """
    Makes observation space correct as well, so the whole interface is correct
    """
    def __init__(self, env):
        super(DMSuiteUnwrapper, self).__init__(env)
        self.observation_space = self.observation_space['observations']

    def reset(self, *args, **kwargs):
        state = super(DMSuiteUnwrapper, self).reset(*args, **kwargs)
        assert isinstance(state, OrderedDict)
        return state['observations']

    def step(self, *args, **kwargs):
        state, reward, done, info = super(DMSuiteUnwrapper,
                                          self).step(*args, **kwargs)
        assert isinstance(state, OrderedDict)
        return state['observations'], reward, done, info

    def render(self, *args, **kwargs):
        kwargs['use_opencv_renderer'] = True
        return super(DMSuiteUnwrapper, self).render(*args, **kwargs)

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip:int=1):
        print('frame skipping!')
        assert isinstance(skip, int) and skip >= 1, f"Skip was {skip}"
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.
        done = False
        for _ in range(self._skip):
            # print('doing skip thing')
            obs, r, done, info = self.env.step(action)
            total_reward += r
            if done:
                break

        return obs, total_reward, done, info

        

def make_env(env_name, step_limit=None, delay_time=-1, action_skip=1, seed=0):
    """
    env_name: gets passed to gym.make
    step_limit: goes into the TimeLimit wrapper
    """
    import rbfdqn.tasks
    if env_name.startswith("dm2gym:"):
        if step_limit is not None:
            raise ValueError("For now, no new step limits with dm_control stuff")
        delay_time_dict = {
        'dm2gym:CartpoleSwingup_sparse-v0': 0,
            'dm2gym:PendulumSwingup-v0': 1000,
            'dm2gym:Ball_in_cupCatch-v0': 1000,
            'dm2gym:AcrobotSwingup_sparse-v0': 1000,
            'dm2gym:HopperStand-v0': 1000,
        }
        if env_name not in delay_time_dict:
            raise Exception(
                f"Currently only can handle these four domains, got {env_name}"
            )
        delay_time = delay_time if delay_time >= 0 else delay_time_dict[env_name]

        env = gym.make(env_name, environment_kwargs={'flat_observation': True}, task_kwargs={"random": seed})
        env = DMSettlingWrapper(env,
                                random_steps=delay_time,
                                reset_on_reward=True)
        env = DMSuiteUnwrapper(env)
    else:
        env = gym.make(env_name)
        if step_limit is not None:
            print(f"Setting step-limit for {env_name} to {step_limit}")
            assert isinstance(env, gym.wrappers.TimeLimit), "for now assume it's time-limit wrapped"
            env = env.unwrapped
            env = TimeLimit(env, max_episode_steps=step_limit)

    if action_skip != 1:
        env = FrameSkipWrapper(env, skip=action_skip)

    return env
