import os
from pathlib import Path

from scipy.spatial.distance import num_obs_dm
from rbfdqn.plotting.state_plotter import make_state_and_exploration_plots
"""
Actual useful file for state plotting
"""


def plot_all_seeds_and_runs(base_dir, point_maze=False, num_episodes=-1):
    """
    Convenience method. Pass in your experiment_name,
    this will iterate through each run_title and seed in it,
    and make a plot
    Arguments:
        base_dir: Something with one or more runs in it
        point_maze: Determines whether or not we should draw the maze lines in background
    """
    directory = Path(base_dir)
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        stored_states_dir = child / "stored_states"
        for new_child in stored_states_dir.iterdir():
            if new_child.is_dir():
                raise Exception("Shouldn't happen!")
            print(str(new_child))
            make_state_and_exploration_plots(str(new_child),
                                             point_maze=point_maze,
                                             num_episodes=num_episodes)


def plot_all_seeds_for_run(dir_name, point_maze=False, num_episodes=-1):
    """
    Plots all seeds for given run.
    Arguments:
        dir_name: The run directory. Should have a "stored_states" subdirectory
        point_maze: Determines whether or not we should draw the maze lines in background
    """
    directory = Path(dir_name)
    states_directory = directory / "stored_states"
    if not states_directory.is_dir():
        raise Exception("Seems like you're missing a states dir")
    for child in states_directory.iterdir():
        print(str(child))
        make_state_and_exploration_plots(str(child), point_maze=point_maze)


if __name__ == '__main__':
    """
    Three examples given below -- uncomment one and edit parameters to suit your needs.
    """
    # base_dir = "./plots/example_experiment"
    # plot_all_seeds_and_runs(base_dir, point_maze=False)

    # dir_name = "./plots/example_experiment/run_title"
    # plot_all_seeds_for_run(dir_name, point_maze=False)

    # pkl_filename = "./plots/example_experiment/run_title/stored_states/seed_0_stored_states.pkl"
    # make_state_and_exploration_plots(pkl_filename, point_maze=False)
