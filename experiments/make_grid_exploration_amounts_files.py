"""
You need to run this file in order to create the grid_exploration_amounts files for plotting.
Requires that you store states during training.
"""

import math
import pickle
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

def discretize_state_list(l, square_size=0.5):
    """
    List of states, "radius"
    """
    discrete_tuples = []
    for state in l:
        t = tuple(math.floor(f/square_size) for f in state)
        discrete_tuples.append(t)
    return discrete_tuples


def test_discretize_example():
    l = [
        [0.1, 0.9],
        [-0.1, 0.1],
        [1.8, -1.1],
    ]
    discretized = discretize_state_list(l, 1.0)
    assert discretized == [
        (0, 0),
        (-1,0),
        (1,-2)
    ]


def add_discretization_to_set(s, l, square_size=0.5):
    discretized = discretize_state_list(l, square_size=square_size)
    s.update(discretized)
    return s

def trim_list_of_episodes(list_of_episodes):
    los = []
    for l in list_of_episodes:
        trimmed_states = [s[0:2] for s in l]
        los.append(trimmed_states)
    return los

def episodes_to_exploration_amounts(list_of_episodes, square_size=0.5):
    """
    list of episodes is a list of lists of states.
    Figure out how much was explored at each step.
    """
    list_of_episodes = trim_list_of_episodes(list_of_episodes)
    visitation_set = set()
    exploration_per_episode = []
    for episode in list_of_episodes:
        add_discretization_to_set(visitation_set, episode, square_size=square_size)
        exploration_per_episode.append(len(visitation_set))

    return exploration_per_episode


def write_grid_exploration_pkl(input_filename, output_filename):
    with open(input_filename, "rb") as f:
        list_of_episodes = pickle.load(f)['states']

    expl_per_ep = episodes_to_exploration_amounts(list_of_episodes, square_size=0.5)

    with open(output_filename, "wb") as f:
        pickle.dump(expl_per_ep, f)

def do_for_all_files_in_run(run_directory):
    print(f'doing for {run_directory}')
    if isinstance(run_directory, str):
        run_directory = Path(run_directory)


    states_directory = run_directory / "stored_states"
    if not states_directory.exists():
        print(f"no states directory for {str(run_directory)}, continuing")
        return

    grid_directory = run_directory / "grid_exploration_amounts"
    grid_directory.mkdir(exist_ok=True)

    for filepath in states_directory.iterdir():
        filename = filepath.name
        gridpath = grid_directory / filename
        write_grid_exploration_pkl(str(filepath), str(gridpath))

    print(f'done for {run_directory}')

def do_for_all_runs_in_experiment(exp_directory):
    print(f'doing for {exp_directory}')
    if isinstance(exp_directory, str):
        exp_directory = Path(exp_directory)
    
    for directory in exp_directory.iterdir():
        if not directory.is_dir():
            continue
        try:
            do_for_all_files_in_run(directory)
        except:
            print(f"failed for {str(directory)}")


def do_for_pickle_file(filename):
    with open(filename, "rb") as f:
        list_of_episodes = pickle.load(f)['states']

    expl_per_ep = episodes_to_exploration_amounts(list_of_episodes, square_size=0.5)
    plt.plot(range(len(expl_per_ep)), np.array(expl_per_ep) / 400)
    plt.show()
    plt.close()


def main():
    do_for_all_runs_in_experiment("./remote_plots/slurm_plots/point_maze/ddpg/no_knownness/fixed_policy")

def tests():
    test_discretize_example()

if __name__ == "__main__":
    # tests()
    main()