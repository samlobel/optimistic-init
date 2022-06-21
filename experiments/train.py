import os, sys, argparse
import pickle
import gym
import numpy as np
from rbfdqn.plotting.learning_curve_plotter import MetaLogger
from rbfdqn import utils_for_q_learning
from rbfdqn import utils
from rbfdqn.RBFDQN import Net, RandomAgent
from rbfdqn.TD3 import TD3Net
from time import time
import datetime
from distutils.util import strtobool
import random
import torch

import rbfdqn.tasks  # NOQA: This makes the environments registerable....

from rbfdqn.exploration_logging_utils import get_coreset_volume
from rbfdqn.exploration import TorchStateActionKnownness
from rbfdqn.scaling_functions import get_scaling_array
from rbfdqn import scaling_functions

from rbfdqn.plotting.state_plotter import XYStatePlotter, EpisodeLogger

import warnings
warnings.simplefilter('error', RuntimeWarning)
"""
This is going to be an experiment that does something like, 
ignores the reward and just writes down the amount of exploration
it does overall.
Maybe, ignore reward will be an OPTION, not a prerequisite.
"""

PARAM_DEFAULTS = {
    "skip_normalization": False,
    "approx_filter_radius": 0.0,
    "policy_type": "e_greedy",
}



def get_coreset_iterative(current_coreset,
                          trajectory,
                          radius=1.,
                          normalizer=1.,
                          only_xy=False,
                          only_states=False):
    """
    only_xy:
        Does the same thing, but only considers the first two dimensions of the state, and not the action.
    """
    # You should be able to just use the previous compression as your starting point...
    assert not (only_xy and only_states), "can't be both!"
    if len(current_coreset) == 0 and len(trajectory) == 0:
        return []

    if only_xy:
        volume_elements = [s[0:2] for (s, a, r, ns, t) in trajectory]
        volume_elements = np.array(volume_elements) / normalizer
    elif only_states:
        volume_elements = [s for (s, a, r, ns, t) in trajectory]
        volume_elements = np.array(volume_elements) / normalizer
    else:
        new_states = []
        new_actions = []
        for state, action, reward, next_state, is_terminal in trajectory:
            new_states.append(state)
            new_actions.append(action)

        new_states = np.array(new_states)
        new_actions = np.array(new_actions)
        sa_concat = np.concatenate((new_states, new_actions), axis=1)

        volume_elements = sa_concat / normalizer

    coreset = get_coreset_volume(volume_elements,
                                 starting_core_set=current_coreset,
                                 radius=radius)

    coreset = np.array(coreset)
    return coreset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_parameter_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)
    parser.add_argument(
        "--run_title", type=str,
        required=True)  # This is the subdir that we'll be saving in.
    # parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--render_every", type=int, default=0)
    parser.add_argument("--ignore_reward", action="store_true", default=False)
    parser.add_argument("--use_exploration",
                        action="store_true",
                        default=False)
    parser.add_argument("--use_counting",
                        action="store_true",
                        default=False)
    parser.add_argument("--use_knownness", action="store_true", default=False)
    parser.add_argument("--use_rnd", action="store_true", default=False)
    parser.add_argument("--use_mpe_bonus", action="store_true", default=False)
    parser.add_argument("--random_agent", action="store_true", default=False)
    parser.add_argument(
        "--evaluation_agent",
        type=utils.boolify,
        default=False,
        help=
        "makes a copy of the network that only trains with extrinsic bonus. Say 'true' or 'false'"
    )

    parser.add_argument("--use_torch_knownness",
                        action="store_true",
                        default=False)

    parser.add_argument("--log_knownness_etc",
                        action="store_true",
                        default=False)

    parser.add_argument("--log_com_knownness",
                        action="store_true",
                        default=False)

    parser.add_argument("--only_state_knownness",
                        action="store_true",
                        default=False)

    parser.add_argument("--device",
                        default="cpu",
                        help="Should be either cpu or cuda")

    parser.add_argument("--visualize_sibriv_expl",
                        action="store_true",
                        default=False)
    parser.add_argument("--visualize_values",
                        action="store_true",
                        default=False,
                        help="More general visualize_sibriv_expl")
    parser.add_argument(
        "--clip_q_targets",
        action="store_true",
        default=False,
        help=
        "Clips q-target to max-q, in situations where it's pushing the estimate above that."
    )
    parser.add_argument(
        "--use_shaping",
        action="store_true",
        default=False,
        help="If you want to use a shaping function, provided one exists.")
    parser.add_argument("--shaping_func_name",
                        default=None,
                        help="Tells it what shaping name to use.")

    parser.add_argument("--store_states",
                        action="store_true",
                        default=False,
                        help="Store states somewhere, per episode.")

    parser.add_argument(
        "--track_xy_exploration",
        default=False,
        action="store_true",
        help=
        "In addition to state/action exploration, track the volume of the first two state-dimensions."
    )

    parser.add_argument(
        "--per_episode_step_limit",
        default=None,
        type=int,
        help=
        "Wraps env in a TimeLimit with this number of steps. Defaults to not doing this"
    )

    parser.add_argument(
        "--delay_time",
        default=-1,
        type=int,
        help=
        "Lets you set arbitrary delay times. Will be helpful for searching for the right delay length."
    )

    parser.add_argument("--evaluations_per_episode",
                        default=5,
                        type=int,
                        help="Number of testing runs per training run. ")

    parser.add_argument("--log_filter_percentage",
                        default=False,
                        action="store_true",
                        help="Logs the amount you're filtering.")

    parser.add_argument("--use_naive_optimism",
                        action="store_true",
                        default=False,
                        help="If set, you boost values by q_max")

    parser.add_argument("--use_TD3",
                        action="store_true",
                        default=False,
                        help="If set, uses TD3 instead of RBFDQN")


    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    use_shaping = args.shaping_func_name is not None and args.shaping_func_name.lower(
    ) != "none"

    assert not (args.random_agent and
                (args.use_knownness
                 or args.use_exploration)), "can't do random with exploration."

    if args.use_torch_knownness:
        assert args.use_knownness, "torch_knownness doesn't make sense without knownness"
    if args.evaluation_agent:
        assert not args.random_agent, "concept of an evaluation agent doesn't make much sense with random..."
        assert args.use_knownness or args.use_exploration, "it doesn't make sense to have an evaluation agent with no exploration algo, since then you could just use the regular q-network"

    if args.only_state_knownness:
        assert args.use_knownness, "can't do state knownness without knownness"

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"
    meta_logger.add_field("scores", logging_filename)
    meta_logger.add_field("exploration_amounts", logging_filename)
    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("training_times", logging_filename)
    meta_logger.add_field("all_times", logging_filename)
    meta_logger.add_field("average_q_buffer", logging_filename)
    meta_logger.add_field("average_q_target_buffer", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)

    if args.use_counting:
        meta_logger.add_field("average_unscaled_count_bonus", logging_filename)

    if args.track_xy_exploration:
        meta_logger.add_field("xy_exploration_amounts", logging_filename)

    if args.log_filter_percentage:
        meta_logger.add_field("filter_percentage", logging_filename)

    if args.store_states:
        episode_logger = EpisodeLogger()
        utils.create_log_dir(
            os.path.join(full_experiment_name, "stored_states"))
        # This one stays because there's a lot of munging.
    else:
        episode_logger = None

    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    print("log dir created?")

    alg = 'rbf'
    params = utils_for_q_learning.get_hyper_parameters(
        args.hyper_parameter_name, alg)

    print("Updating params early now")
    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    params['hyper_parameters_name'] = args.hyper_parameter_name
    env = utils.make_env(params['env_name'],
                         step_limit=args.per_episode_step_limit,
                         delay_time=args.delay_time,
                         action_skip=params['frame_skip'],
                         seed=args.seed)
    params['env'] = env
    params['seed_number'] = args.seed
    params[
        'per_episode_step_limit'] = args.per_episode_step_limit  # fine if this is None, won't be written to hyper

    params['start_time'] = str(datetime.datetime.now())

    if use_shaping:
        params['shaping_func_name'] = args.shaping_func_name

    assert params["knownness_mapping_type"] in [
        "exponential", "normal", "polynomial", "hard", "one_over_x_plus_one"
    ], '"{}" must be one of ["exponential", "normal", "polynomial", "hard"]'.format(
        params["knownness_mapping_type"])

    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string

    params['hyperparams_dir'] = hyperparams_dir
    utils_for_q_learning.save_hyper_parameters(params, args.seed)

    # If filter radius is greater than 0., we assume that you want to use it.
    use_approx_knownness = (params.get(
        "approx_filter_radius", PARAM_DEFAULTS["approx_filter_radius"]) != 0)
    if not args.use_knownness:
        print(
            "Approx knownness set (maybe from hyper file) but not used because no knownness this run"
        )
        use_approx_knownness = False

    utils_for_q_learning.set_random_seed(params)

    if args.visualize_sibriv_expl:
        assert params[
            'env_name'] == 'SibRivPointMaze-v1', "can't visualize sibriv if it's not sibriv!!!"

    # Get your knownness scaling array
    knownness_scaling_array = None
    if "env_bound_scaling_string" in params:
        knownness_scaling_array = params['env_bound_scaling_string'].split(
            "__")
        knownness_scaling_array = [
            float(scaler) for scaler in knownness_scaling_array
        ]
        knownness_scaling_array = np.array(knownness_scaling_array)
    if "action_scaling" in params:
        action_scaling_array = scaling_functions.action_scaling(
            env, params['action_scaling'])
        if knownness_scaling_array is None:
            knownness_scaling_array = action_scaling_array
        else:
            knownness_scaling_array = knownness_scaling_array * action_scaling_array

    s0 = env.reset()
    utils_for_q_learning.action_checker(env)

    if args.use_TD3:
        NetConstructor = TD3Net
    else:
        NetConstructor = Net

    if args.random_agent:
        Q_object = RandomAgent(params,
                               env,
                               state_size=len(s0),
                               action_size=len(env.action_space.low))
        Q_object_target = RandomAgent(params,
                                      env,
                                      state_size=len(s0),
                                      action_size=len(env.action_space.low))
    else:
        Q_object = NetConstructor(
            params,
            env,
            state_size=len(s0),
            action_size=len(env.action_space.low),
            use_exploration=args.use_exploration,
            use_counting=args.use_counting,
            use_rnd=args.use_rnd,
            use_mpe_bonus=args.use_mpe_bonus,
            use_knownness=args.use_knownness,
            use_torch_knownness=args.use_torch_knownness,
            only_state_knownness=args.only_state_knownness,
            clip_q_targets=args.clip_q_targets,
            knownness_mapping_type=params["knownness_mapping_type"],
            use_shaping=use_shaping,
            shaping_func_name=args.shaping_func_name,
            skip_exploration_normalization=params.get(
                "skip_normalization", PARAM_DEFAULTS["skip_normalization"]),
            knownness_scaling_array=knownness_scaling_array,
            use_approx_knownness=use_approx_knownness,
            approx_filter_radius=params.get(
                "approx_filter_radius",
                PARAM_DEFAULTS["approx_filter_radius"]),
            device=args.device,
            use_naive_optimism=args.use_naive_optimism,
        )
        Q_object_target = NetConstructor(
            params,
            env,
            state_size=len(s0),
            action_size=len(env.action_space.low),
            use_exploration=False,
            use_counting=False,
            use_rnd=False,
            use_mpe_bonus=False,
            use_knownness=False,
            clip_q_targets=args.clip_q_targets,
            knownness_mapping_type=params["knownness_mapping_type"],
            use_shaping=use_shaping,
            shaping_func_name=args.shaping_func_name,
            skip_exploration_normalization=params.get(
                "skip_normalization", PARAM_DEFAULTS["skip_normalization"]),
            device=args.device,
            use_naive_optimism=args.use_naive_optimism,
        )
        Q_object_target.eval()

        utils_for_q_learning.sync_networks(
            target=Q_object_target,
            online=Q_object,
            alpha=params['target_network_learning_rate'],
            copy=True)

    if args.evaluation_agent:
        print("We're going to be making an evaluation agent")
        eval_Q_object = NetConstructor(
            params,
            env,
            state_size=len(s0),
            action_size=len(env.action_space.low),
            use_exploration=False,
            use_rnd=False,
            use_knownness=False,
            device=args.device,
        )  # The whole point is that we don't do knownness stuff now.
        eval_Q_object_target = NetConstructor(
            params,
            env,
            state_size=len(s0),
            action_size=len(env.action_space.low),
            use_exploration=False,
            use_rnd=False,
            use_knownness=False,
            device=args.device,
        )
        eval_Q_object_target.eval()

        utils_for_q_learning.sync_networks(
            target=eval_Q_object_target,
            online=Q_object_target,
            alpha=params['target_network_learning_rate'],
            copy=True)
        utils_for_q_learning.sync_networks(
            target=eval_Q_object,
            online=Q_object_target,
            alpha=params['target_network_learning_rate'],
            copy=True)

    EVAL_AGENT = eval_Q_object if args.evaluation_agent else Q_object

    volume_normalizer = knownness_scaling_array[0:Q_object.
                                                state_size]**-1  # 1 over it.

    current_coreset = []

    if args.track_xy_exploration:
        current_xy_coreset = []
        meta_logger.add_field("xy_exploration_amounts", logging_filename)

    if args.visualize_sibriv_expl:
        sibriv_visualizer = XYStatePlotter(env=env)
        utils.create_log_dir(
            os.path.join(full_experiment_name, "sibriv_states"))
        utils.create_log_dir(os.path.join(full_experiment_name,
                                          "sibriv_plots"))
        sibriv_states_filename = os.path.join(
            full_experiment_name, "sibriv_states",
            "seed_" + str(args.seed) + "_sibriv_states.pkl")
        sibriv_q_w_novelty_filename = os.path.join(
            full_experiment_name, "sibriv_plots",
            "seed_" + str(args.seed) + "_episode_{}_sibriv_q_with_novelty.jpg")
        sibriv_q_wo_novelty_filename = os.path.join(
            full_experiment_name, "sibriv_plots", "seed_" + str(args.seed) +
            "_episode_{}_sibriv_q_without_novelty.jpg")
        sibriv_quiver_plot_filename = os.path.join(
            full_experiment_name, "sibriv_plots",
            "seed_" + str(args.seed) + "_episode_{}_sibriv_quiver_plot.svg")

    if args.visualize_values:
        value_visualizer = XYStatePlotter(env=env)
        values_dir = os.path.join(full_experiment_name, "value_plots")
        os.makedirs(values_dir, exist_ok=True)
        q_w_novelty_filename = os.path.join(
            values_dir,
            "seed_" + str(args.seed) + "_episode_{}_q_with_novelty.pkl")
        q_wo_novelty_filename = os.path.join(
            values_dir,
            "seed_" + str(args.seed) + "_episode_{}_q_without_novelty.pkl")
        quiver_plot_filename = os.path.join(
            values_dir,
            "seed_" + str(args.seed) + "_episode_{}_quiver_plot.pkl")

    else:
        sibriv_visualizer = None

    # This one stays for now because there's lots of munging.
    stored_states_filename = os.path.join(
        full_experiment_name, "stored_states",
        "seed_" + str(args.seed) + "_stored_states.pkl")

    TOTAL_STATES_VISITED = 0
    for episode in range(params['max_episode']):
        episode_start_time = time()
        #train policy with exploration
        s, done = env.reset(), False
        episodic_trajectory = []
        episodic_reward = 0.
        start_time = time()
        print("Doing training run now...")
        t = 0
        while done == False:
            TOTAL_STATES_VISITED += 1
            if args.render_every > 0 and t % args.render_every == 0:
                env.render()
            policy_type = params.get('policy_type',
                                     PARAM_DEFAULTS['policy_type'])
            a = Q_object.enact_policy(
                s,
                episode + 1,
                'train',
                policy_type=policy_type,
                use_exploration_if_enabled=(Q_object.params["bootstrap_counts"] if args.use_counting else True)
            )

            sp, r, done, info = env.step(np.array(a))
            episodic_reward += r
            if args.ignore_reward:
                r = 0.

            done_for_buffer = done and not info.get('TimeLimit.truncated',
                                                    False)
            if done_for_buffer:
                print(
                    'done in a good way this time. Should only print with high reward for now.'
                )
            episodic_trajectory.append((s, a, r, sp, done_for_buffer))
            s = sp
            t += 1
        print("Episode {} interaction took {:10.4f} seconds ({} steps)".format(
            episode,
            time() - start_time, len(episodic_trajectory)))

        meta_logger.append_datapoint("episodic_rewards",
                                     episodic_reward,
                                     write=True)

        if args.visualize_sibriv_expl:
            sibriv_visualizer.add_episode(episodic_trajectory)
            sibriv_visualizer.write_episodes(sibriv_states_filename)
            if not args.random_agent:
                sibriv_visualizer.write_q_value_plots(
                    Q_object,
                    show=False,
                    save_path=sibriv_q_w_novelty_filename.format(episode))
                sibriv_visualizer.write_q_value_plots(
                    Q_object,
                    show=False,
                    use_novelty=False,
                    save_path=sibriv_q_wo_novelty_filename.format(episode))
                sibriv_visualizer.write_quiver_plots(
                    Q_object,
                    show=False,
                    save_path=sibriv_quiver_plot_filename.format(episode))

        if args.visualize_values:
            if not args.random_agent:
                print('visualizing values')
                value_visualizer.write_q_value_numbers(
                    Q_object,
                    use_novelty=True,
                    save_path=q_w_novelty_filename.format(episode))
                value_visualizer.write_q_value_numbers(
                    Q_object,
                    use_novelty=False,
                    save_path=q_wo_novelty_filename.format(episode))
                value_visualizer.write_quiver_plot_numbers(
                    Q_object,
                    use_novelty=False,
                    save_path=quiver_plot_filename.format(episode))

        if args.store_states:
            episode_logger.add_episode(episodic_trajectory)
            episode_logger.write_episodes(stored_states_filename)

        # Update buffer
        start_time = time()
        Q_object.add_trajectory_to_replay_buffer(episodic_trajectory)
        print("Episode {} replay-buffer-add took {:10.4f} seconds ({} steps)".
              format(episode,
                     time() - start_time, len(episodic_trajectory)))

        if args.log_filter_percentage:
            states_in_novelty = len(
                Q_object.novelty_tracker.unnormalized_buffer)
            print(
                f"{states_in_novelty}/{TOTAL_STATES_VISITED} states in novelty vs visited"
            )
            meta_logger.append_datapoint(
                "filter_percentage",
                (states_in_novelty / TOTAL_STATES_VISITED),
                write=True)

        #now update the Q network
        start_time = time()
        updates_per_episode = params['updates_per_episode']

        average_q_buffer = 0.
        average_q_target_buffer = 0.
        average_loss = 0.
        if args.use_counting:
            average_unscaled_count_bonus = 0

        for _ in range(updates_per_episode):
            update_info = Q_object.update(Q_object_target,
                                          return_logging_info=True)
            if update_info:
                average_q_buffer += update_info['average_q']
                average_q_target_buffer += update_info['average_q_target']
                average_loss += update_info['loss']
                if 'average_unscaled_count_bonus' in update_info:
                    average_unscaled_count_bonus += update_info['average_unscaled_count_bonus']
        average_q_buffer = average_q_buffer / updates_per_episode
        average_q_target_buffer = average_q_target_buffer / updates_per_episode
        average_loss = average_loss / updates_per_episode
        meta_logger.append_datapoint("average_q_buffer",
                                     average_q_buffer,
                                     write=True)
        meta_logger.append_datapoint("average_q_target_buffer",
                                     average_q_target_buffer,
                                     write=True)
        meta_logger.append_datapoint("average_loss", average_loss, write=True)
        if args.use_counting:
            average_unscaled_count_bonus = average_unscaled_count_bonus / updates_per_episode
            meta_logger.append_datapoint("average_unscaled_count_bonus", average_unscaled_count_bonus, write=True)

        this_training_time = time() - start_time
        meta_logger.append_datapoint("training_times",
                                     this_training_time,
                                     write=True)

        print("Batch training took ", "{:10.4f}".format(this_training_time),
              " to perform ", params['updates_per_episode'], " updates")

        if args.evaluation_agent:
            start_time = time()
            eval_Q_object.buffer_object = Q_object.buffer_object
            for _ in range(params['updates_per_episode']):
                eval_Q_object.update(eval_Q_object_target)
            print("Eval training took {:10.4f} seconds".format(time() -
                                                               start_time))

        evaluation_return = []
        total_steps = 0
        start_time = time()
        print("Doing eval runs now...")
        for _ in range(args.evaluations_per_episode):
            s, t, G, done = env.reset(), 0, 0, False
            while done == False:
                if args.render_every > 0 and t % args.render_every == 0:
                    env.render()
                a = EVAL_AGENT.e_greedy_policy(
                    s, episode + 1, 'test', use_exploration_if_enabled=False
                )  # No exploration during eval...
                sp, r, done, _ = env.step(np.array(a))
                total_steps += 1
                G += r
                s = sp
            evaluation_return.append(G)

        average_evaluation_return = np.mean(evaluation_return)
        print("Episode {}: {} evaluation{} took {:10.4f} seconds ({} steps)".
              format(episode, args.evaluations_per_episode,
                     "" if args.evaluations_per_episode == 1 else "s",
                     time() - start_time, total_steps))
        print("in episode {} we collected average return {}".format(
            episode, average_evaluation_return))
        meta_logger.append_datapoint("scores",
                                     average_evaluation_return,
                                     write=True)

        meta_logger.append_datapoint("all_times",
                                     time() - episode_start_time,
                                     write=True)

        start_time = time()
        current_coreset = get_coreset_iterative(
            current_coreset,
            episodic_trajectory,
            radius=params['volume_counting_radius'],
            normalizer=volume_normalizer,
            only_states=True)

        coreset_length = len(current_coreset)
        meta_logger.append_datapoint("exploration_amounts",
                                     coreset_length,
                                     write=True)

        print("Coreset length: ", coreset_length, " from: ",
              len(Q_object.buffer_object))
        print(
            "Episode {} coreset stuff took {:10.4f} seconds ({} steps)".format(
                episode,
                time() - start_time, t))

        if args.track_xy_exploration:
            current_xy_coreset = get_coreset_iterative(current_xy_coreset,
                                                       episodic_trajectory,
                                                       radius=1.,
                                                       normalizer=1.,
                                                       only_xy=True)

            xy_coreset_length = len(current_xy_coreset)
            meta_logger.append_datapoint("xy_exploration_amounts",
                                         xy_coreset_length,
                                         write=True)
