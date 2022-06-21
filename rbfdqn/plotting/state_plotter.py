"""
Previously known as "sibriv-logger", this ports the useful code. Then,
in experiments we will have something that just imports the bits we'd use.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import gym
from rbfdqn import RBFDQN


from contextlib import contextmanager

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

class EpisodeLogger:
    """
    Mostly cribbed from the XYStatePlotter, this is just simpler.
    TODO: Make this have an interface more like MetaLogger.
    """
    def __init__(self):
        self.episode_states = []
        self.episode_actions = []

    def add_episode(self, episodic_trajectory):
        states = np.array([t[0] for t in episodic_trajectory])
        actions = np.array([t[1] for t in episodic_trajectory])
        self.episode_states.append(states)
        self.episode_actions.append(actions)

    def write_episodes(self, logfile):
        sa = {'states': self.episode_states, 'actions': self.episode_actions}
        with open(logfile, "wb") as f:
            # pickle.dump(self.episode_states, f)
            pickle.dump(sa, f)


class XYStatePlotter:
    """
    Pretty much, we'll be doing "add_episode" or whatever.
    We'll keep them all separate as states. We'll also keep the env around.
    Then, we can make plots that show the progress so far.

    I wanted to do something like make a novelty-detector on the fly, but I'm not sure I'll be able to
    without actions. Maybe I'll just add the actions.

    """
    def __init__(self, env : gym.Env =None):
        self.env = env
        self.episode_states = []
        self.episode_actions = []

    def add_episode(self, episodic_trajectory):
        states = np.array([t[0] for t in episodic_trajectory])
        actions = np.array([t[1] for t in episodic_trajectory])
        self.episode_states.append(states)
        self.episode_actions.append(actions)

    def make_animation(self, show=True, save_path=None, episode_states=None):
        fig, ax = plt.subplots()

        pass

    def plot_stuff_so_far(self,
                          show=True,
                          save_path=None,
                          episode_states=None,
                          num_episodes=-1,
                          ax=None):
        # ax: matplotlib axis.
        if episode_states is None:
            episode_states = self.episode_states
        if num_episodes >= 0:
            episode_states = episode_states[0:num_episodes]

        all_states = [s for ep in episode_states for s in ep]
        if all_states and len(all_states[0]) > 2:
            print(f"What is dimension? {all_states[0]}")
            print(
                "Need to filter dimensions, because we can only plot two. Filtering to first two."
            )
            # all_states = [s[0:2] for s in all_states]
            # all_states = [np.array([s[0],s[2]]) for s in all_states]
            print("Just for acrobot")
            all_states = [np.array([s[0]+s[1], s[2]+s[3]]) for s in all_states]
        all_colors = [i for i, ep in enumerate(episode_states)
                      for s in ep]  # color for each ep
        all_states_x, all_states_y = list(zip(*all_states))

        plt.close()
        try:
            self.env.maze.plot(ax=ax)
        except:
            print(
                "We're not going to be able to do that when it's from planeworld."
            )

        if ax:
            plt.scatter(all_states_x, all_states_y, c=all_colors)
            plt.colorbar()
            # plt.title("Everywhere agent has been, per episode")
            plt.title("OURS")
        else:
            plt.scatter(all_states_x, all_states_y, c=all_colors)
            plt.colorbar()
            plt.title("OURS")

        if show:
            plt.show()
        if save_path is not None:
            print('saving to ', save_path)
            plt.savefig(save_path)

    def plot_stuff_from_log(self,
                            logfile,
                            show=True,
                            save_path=None,
                            num_episodes=-1):
        with open(logfile, "rb") as f:
            sa = pickle.load(f)
            try:
                states, actions = sa['states'], sa['actions']
            except:
                states, actions = sa, None  # If it's the old way.
            # episode_states = pickle.load(f)
        self.plot_stuff_so_far(show=show,
                               episode_states=states,
                               num_episodes=num_episodes,
                               save_path=save_path)

    def write_episodes(self, logfile):
        sa = {'states': self.episode_states, 'actions': self.episode_actions}
        with open(logfile, "wb") as f:
            # pickle.dump(self.episode_states, f)
            pickle.dump(sa, f)

    def write_q_value_numbers(self, q_agent: RBFDQN.Net, use_novelty=False, save_path=None):
        # For now, 2d because of viz limitations.
        assert self.env.observation_space.shape == (2, )
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        x = np.linspace(obs_low[0], obs_high[0], 100)
        y = np.linspace(obs_low[1], obs_high[1], 100)
        xx, yy = np.meshgrid(x, y)
        xxf = xx.reshape(-1, 1)
        yyf = yy.reshape(-1, 1)
        states = np.hstack((xxf, yyf))

        with torch.no_grad():
            novelty_tracker = q_agent.novelty_tracker if (
                q_agent.use_knownness and use_novelty) else None
            with evaluating(q_agent):
                q_star, _ = q_agent.get_best_qvalue_and_action(
                    torch.FloatTensor(states).to(q_agent.device), novelty_tracker=novelty_tracker, use_exploration_if_enabled=use_novelty)

        q_star = q_star.cpu().numpy()
        to_write = (xxf.reshape(-1), yyf.reshape(-1), q_star.reshape(-1))
        with open(save_path, "wb") as f:
            pickle.dump(to_write, f)
        
        print('dumped!')
        return
            

    def write_quiver_plot_numbers(self, q_agent: RBFDQN.Net, use_novelty=False, save_path=None):
        assert self.env.observation_space.shape == (2, )
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        x = np.linspace(obs_low[0], obs_high[0], 40)
        y = np.linspace(obs_low[1], obs_high[1], 40)
        xx, yy = np.meshgrid(x, y)
        xxf = xx.reshape(-1, 1)
        yyf = yy.reshape(-1, 1)
        states = np.hstack((xxf, yyf))


        with torch.no_grad():
            novelty_tracker = q_agent.novelty_tracker if (
                q_agent.use_knownness and use_novelty) else None
            with evaluating(q_agent):
                q_star, best_actions = q_agent.get_best_qvalue_and_action(
                    torch.FloatTensor(states).to(q_agent.device), novelty_tracker=novelty_tracker, use_exploration_if_enabled=use_novelty, return_batch_action=True)
        
        
        q_star = q_star.cpu().numpy()
        best_actions = best_actions.cpu().numpy()

        to_write = (xxf.reshape(-1), yyf.reshape(-1), best_actions, q_star.reshape(-1))
        with open(save_path, "wb") as f:
            pickle.dump(to_write, f)
        

    def make_q_value_plot_from_log(self, load_path, save=False):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        x, y, v = data
        plt.scatter(x, y, c=v)
        plt.colorbar()
        plt.show()

    def make_quiver_plot_from_log(self, load_path, save=False):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        x, y, a, v = data
        # import ipdb; ipdb.set_trace()
        first_a = a[:,0]
        if a.shape[1] > 1:
            second_a = a[:,1]
        else:
            second_a = np.zeros_like(first_a)

        # for mcar specifically.
        plt.quiver(x, y, first_a, second_a, v)
        # plt.scatter(x, y, c=v)
        plt.set_cmap('cool')
        plt.colorbar()
        plt.show()


    def write_q_value_plots(self,
                            q_agent,
                            use_novelty=True,
                            show=False,
                            save_path=None):
        """
        This one is going to change every episode -- that means that we need to maybe log the
        actual plots, instead of just the data that makes them. Annoying, but such is life.

        I'm going to use the knownness thing in getting the Q value, because that's what we're using for decision-making.
        
        It might be good to actually make this and the episodic one happen on the same graph. That would do a good job
        at making me understand where should be high-value and where should be low-value.

        """

        # 256 is standard batch size so this is a lot already, maybe I'll chunk it?
        x = np.linspace(-0.5, 9.5, 100)
        y = np.linspace(-0.5, 9.5, 100)
        xx, yy = np.meshgrid(x, y)
        xxf = xx.reshape(-1, 1)
        yyf = yy.reshape(-1, 1)
        states = np.hstack((xxf, yyf))

        with torch.no_grad():
            novelty_tracker = q_agent.novelty_tracker if (
                q_agent.use_knownness and use_novelty) else None
            q_star = q_agent.get_best_centroid_batch(
                torch.FloatTensor(states), novelty_tracker=novelty_tracker)

        # plt.clf()
        self.env.maze.plot()
        plt.scatter(states[:, 0], states[:, 1], c=q_star)
        plt.colorbar()
        if use_novelty:
            plt.title(
                "Q values (best action, w. action-sel-bonus) throughout maze")
        else:
            plt.title(
                "Q values (best action, NO action-sel-bonus) throughout maze")
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    def write_quiver_plots(self, q_agent, show=False, save_path=None):
        """
        This one is interesting -- it should give us some insight into the decision-making process of the agent.
        We're going to get the DIRECTION of the best action, for all actions.
        """
        x = np.linspace(-0.5, 9.5, 40)
        y = np.linspace(-0.5, 9.5, 40)
        xx, yy = np.meshgrid(x, y)
        xxf = xx.reshape(-1, 1)
        yyf = yy.reshape(-1, 1)
        states = np.hstack((xxf, yyf))

        with torch.no_grad():
            novelty_tracker = q_agent.novelty_tracker if q_agent.use_knownness else None
            q_star, a_star = q_agent.get_best_centroid_batch(
                torch.FloatTensor(states),
                novelty_tracker=novelty_tracker,
                return_actions=True)

        plt.close()
        self.env.maze.plot()
        # import ipdb; ipdb.set_trace()
        plt.quiver(states[:, 0], states[:, 1], a_star[:, 0], a_star[:, 1],
                   q_star)
        plt.colorbar()
        plt.title("Action directions throughout maze")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


import rbfdqn.tasks
from rbfdqn.exploration import TorchStateActionKnownness, StateActionKnownness


def make_state_and_exploration_plots(logfile,
                                     m=1,
                                     point_maze=False,
                                     save_path=None,
                                     num_episodes=-1):
    """
    How about I do like 10 actions for each state, and then take the max of them
    or something? That will be the MAXIMAL knownness at each state. Which is a lot better than
    knownness of the zero-action. Not too bad of an idea.
    """
    EPSILON = 0.2
    MAPPING_TYPE = "normal"
    import gym
    import rbfdqn.tasks
    env = gym.make("SibRivPointMaze-v1") if point_maze else gym.make(
        "PlaneWorld-v1")

    # import ipdb; ipdb.set_trace()

    plt.clf()
    logger = XYStatePlotter(env=env)
    logger.plot_stuff_from_log(logfile, show=True, save_path=save_path, num_episodes=num_episodes)
    plt.close()

    return

    raise Exception("Old code, doesn't work everywhere, port as needed.")

    with open(logfile, "rb") as f:
        sa = pickle.load(f)

    states, actions = sa['states'], sa['actions']

    states = np.array(states).reshape(-1, 2)
    actions = np.array(actions).reshape(-1, 2)

    novelty_tracker = TorchStateActionKnownness(m=m,
                                                epsilon=EPSILON,
                                                action_scaling=0.1,
                                                mapping_type=MAPPING_TYPE)
    # novelty_tracker = TorchStateActionKnownness(m=m, epsilon=0.4, action_scaling=0.1)
    # novelty_tracker = StateActionKnownness(m=m, epsilon=0.5, action_scaling=0.1)
    # import ipdb; ipdb.set_trace()
    novelty_tracker.add_many_transitions(np.array(states), np.array(actions))

    x = np.linspace(-0.5, 9.5, 100)
    y = np.linspace(-0.5, 9.5, 100)
    xx, yy = np.meshgrid(x, y)
    xxf = xx.reshape(-1, 1)
    yyf = yy.reshape(-1, 1)
    states = np.hstack((xxf, yyf))
    actions = np.zeros_like(states)
    # actions = actions.reshape(-1,1,2)

    knownness = novelty_tracker.get_knownness(states, actions)
    knownness = knownness.reshape(-1)

    # knownness = np.floor(knownness * 255)

    # import ipdb; ipdb.set_trace()

    env.maze.plot()
    # import ipdb; ipdb.set_trace()
    plt.scatter(states[:, 0], states[:, 1], c=knownness)
    # plt.scatter(states[:,0], states[:,1])
    plt.colorbar()
    plt.title("Knownness of zero-action throughout maze")

    plt.show()
    plt.close()

    logger = XYStatePlotter(env=env)
    logger.plot_stuff_from_log(logfile, show=True)
    plt.close()
