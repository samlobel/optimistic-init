from gym.envs.registration import register


register(
    id='SibRivPointMaze-v1',  #120.hyper
    entry_point='rbfdqn.tasks.sibrivmaze.maze_env:Env',
    max_episode_steps=
    50,  # From SibRiv paper... makes it hard cause best policy is like 20 I think.
    reward_threshold=-3.75,  # This just doesn't matter.
    kwargs={
        'n':
        1000,  # This is how they do time-limit-truncation, but I let Gym handle that.
        'fixed_goal': (9., 9.)
    }
)