import gym, sys
import numpy, random

from numpy.lib.financial import _ipmt_dispatcher
from rbfdqn import exploration
from rbfdqn.replay_buffer import buffer_class
from rbfdqn import utils_for_q_learning
from rbfdqn.exploration import (
    DiscretizedStateActionCountingBonus, OnlyStateExplorationClass, StateKnownnessFromMthNeighbor,
    StateActionKnownness, TorchStateActionKnownness,
    TorchNaiveStateKnownness, StateKnownnessFromMthNeighbor,
    TorchStateActionApproxKnownness, StateActionApproxKnownness,
    RNDExploration, MPEExploration)

from rbfdqn.shaping_functions import ShapingFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

EXPL_M = 1

class TD3Net(nn.Module):
    def __init__(self,
                 params,
                 env,
                 state_size,
                 action_size,
                 use_exploration=False,
                 use_rnd=False,
                 use_mpe_bonus=False,
                 use_knownness=False,
                 use_counting=False,
                 use_torch_knownness=False,
                 only_state_knownness=False,
                 clip_q_targets=False,
                 knownness_mapping_type="exponential",
                 use_shaping=False,
                 shaping_func_name=None,
                 skip_exploration_normalization=False,
                 knownness_scaling_array=None,
                 use_approx_knownness=False,
                 approx_filter_radius=0.,
                 use_naive_optimism=False,
                 device="cpu"):
        """
        clip_q_targets: Before combining with knownness, clips targets to q_max. I don't know why they're ever bigger than this,
        but it's some sort of over-estimation I believe.
        """
        super(TD3Net, self).__init__()

        self.device = device

        self.env = env
        self.params = params
        self.N = self.params['num_points']
        assert len(
            set(self.env.action_space.high[0].reshape(-1))
        )  # Make sure all the actions are the same scale, otherwise bug below.
        self.max_a = torch.FloatTensor(self.env.action_space.high).to(
            self.device)
        self.max_a_number = self.env.action_space.high[0]
        self.beta = self.params['temperature']
        num_exploration_modules_loaded = len(list(filter(None, [use_exploration, use_knownness, use_rnd, use_mpe_bonus])))
        assert num_exploration_modules_loaded <= 1, f"Can't load more than one exploration module, loaded {num_exploration_modules_loaded}"
        self.use_exploration = use_exploration
        self.use_rnd = use_rnd
        self.use_mpe_bonus = use_mpe_bonus
        self.use_knownness = use_knownness
        self.use_counting = use_counting
        self.use_torch_knownness = use_torch_knownness
        self.only_state_knownness = only_state_knownness
        self.clip_q_targets = clip_q_targets
        self.knownness_mapping_type = knownness_mapping_type
        self.use_shaping = use_shaping
        self.shaping_func_name = shaping_func_name
        self.use_naive_optimism = use_naive_optimism



        if self.use_shaping:
            self.shaping_class = ShapingFunctions(
                env_name=params['env_name'],
                gamma=params['gamma'],
                func_name=self.shaping_func_name)

        self.normalize_exploration = not skip_exploration_normalization

        self.buffer_object = buffer_class.buffer_class(
            max_length=self.params['max_buffer_size'], env=self.env)

        self.novelty_tracker = None  # In case we access it at some point.
        self.counting_module = None

        if self.use_exploration:
            self.novelty_tracker = OnlyStateExplorationClass(
                epsilon=self.params['counting_epsilon'],
                normalize=True,
                scaling=self.params["reward_clip"])

        if self.use_knownness:
            action_scaling = params.get('action_scaling', 1.0)
            if self.use_torch_knownness:
                if self.only_state_knownness:
                    self.novelty_tracker = TorchNaiveStateKnownness(
                        m=EXPL_M,
                        epsilon=self.params['counting_epsilon'],
                        mapping_type=knownness_mapping_type,
                        normalize=self.normalize_exploration,
                    )
                else:
                    knownness_class = TorchStateActionApproxKnownness if use_approx_knownness else TorchStateActionKnownness
                    self.novelty_tracker = knownness_class(
                        m=EXPL_M,
                        epsilon=self.params['counting_epsilon'],
                        mapping_type=knownness_mapping_type,
                        normalize=self.normalize_exploration,
                        knownness_scaling_array=knownness_scaling_array,
                        filter_radius=approx_filter_radius, #accepts it either way
                    )  # leave batch_size at 100 for now.
            else:
                if self.only_state_knownness:
                    self.novelty_tracker = StateKnownnessFromMthNeighbor(
                        m=EXPL_M,
                        epsilon=self.params['counting_epsilon'],
                        # normalize=True,
                        mapping_type=knownness_mapping_type,
                        normalize=self.normalize_exploration,
                    )
                else:
                    knownness_class = StateActionApproxKnownness if use_approx_knownness else StateActionKnownness
                    self.novelty_tracker = knownness_class(
                        m=EXPL_M,
                        epsilon=self.params['counting_epsilon'],
                        mapping_type=knownness_mapping_type,
                        normalize=self.normalize_exploration,
                        knownness_scaling_array=knownness_scaling_array,
                        filter_radius=approx_filter_radius, # accepts it either way
                        )

        if self.use_counting:
            assert "counting_scaling" in self.params
            self.counting_module = DiscretizedStateActionCountingBonus(
                m=EXPL_M,
                epsilon=self.params['counting_epsilon'],
                normalize=self.normalize_exploration,
                knownness_scaling_array=knownness_scaling_array,
            )
        if self.use_rnd:
            assert 'rnd_scaling' in self.params
            self.rnd_module = RNDExploration(state_dim=state_size, device=self.device)
        if self.use_mpe_bonus:
            assert 'mpe_scaling' in self.params
            self.mpe_module = MPEExploration(
                state_dim=state_size,
                action_dim=action_size,
                use_reward_normalization=params["use_mpe_normalizer"],
                predict_residual=True,
                device=self.device)

        if self.params.get('q_max') is not None:
            self.q_max = self.params['q_max']
        else:
            self.q_max = self.params['reward_clip'] / (1 -
                                                       self.params['gamma'])

        self.state_size, self.action_size = state_size, action_size

        self.value_module_1 = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], 1),
        )

        self.value_module_2 = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], 1),
        )

        self.policy_module = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.action_size),
            nn.Tanh()
        )

        self.criterion = nn.MSELoss()

        # Warning: needs to happen BEFORE optimizer
        # https://pytorch.org/docs/stable/optim.html#constructing-it
        self.to(self.device)

        try:
            if self.params['optimizer'] == 'RMSprop':
                self.policy_optimizer = optim.RMSprop(self.policy_module.parameters(), self.params['learning_rate_policy_module'])
                self.value_optimizer = optim.RMSprop(
                    list(self.value_module_1.parameters()) + list(self.value_module_2.parameters()),
                    self.params['learning_rate'])
            elif self.params['optimizer'] == 'Adam':
                self.policy_optimizer = optim.Adam(self.policy_module.parameters(), self.params['learning_rate_policy_module'])
                self.value_optimizer = optim.Adam(
                    list(self.value_module_1.parameters()) + list(self.value_module_2.parameters()),
                    self.params['learning_rate'])
            else:
                print('unknown optimizer ....')
        except Exception as e:
            print("no optimizer specified ... ")
            raise e

    def forward(self, s, a):
        sa_concat = torch.cat([s, a], dim=1)
        assert sa_concat.shape[-1] == self.state_size + self.action_size
        output_1 = self.value_module_1(sa_concat)
        output_2 = self.value_module_2(sa_concat)
        assert len(output_1.shape) == len(output_2.shape) == 2
        if self.use_naive_optimism:
            output_1 = output_1 + self.q_max
            output_2 = output_2 + self.q_max
        return output_1, output_2


    def get_best_qvalue_and_action(self,
                                   s,
                                   novelty_tracker=None,
                                   use_exploration_if_enabled=True,
                                   return_batch_action=False,
                                   add_action_noise=False):
        '''
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        return_batch_action is False by default because it may be slow and its only used for logging

        I guess I can just do the min here actually. Just used for update, where I do it, and for acting,
        where it doesn't matter. Sure. 
        '''
        best_actions = self.policy_module(s) * self.max_a
        if add_action_noise:
            best_actions += (torch.randn_like(best_actions) * self.params["policy_noise"]).clamp(-0.5, 0.5)
            best_actions = best_actions.clamp(-self.max_a_number, self.max_a_number)
        best_qvalue_1, best_qvalue_2 = self.forward(s, best_actions)

        best_qvalue = torch.min(best_qvalue_1, best_qvalue_2)

        if self.use_shaping:
            upper_bound = self.shaping_class.get_values(s)
            upper_bound = upper_bound.view(-1, 1)
        else:
            upper_bound = self.q_max

        if self.clip_q_targets:

            if isinstance(upper_bound, torch.Tensor):
                best_qvalue = torch.min(best_qvalue, upper_bound)
            elif isinstance(upper_bound, np.ndarray):
                best_qvalue = torch.min(best_qvalue, torch.Tensor(upper_bound))
            elif isinstance(upper_bound, (float, int)):
                best_qvalue = torch.clamp(best_qvalue, max=upper_bound)

        if novelty_tracker is not None and use_exploration_if_enabled and not isinstance(novelty_tracker, DiscretizedStateActionCountingBonus):
            knownnesses = novelty_tracker.get_knownness(s.detach().cpu().numpy(), best_actions.detach().cpu().numpy())
            knownnesses = torch.FloatTensor(knownnesses).to(self.device)
            knownnesses = knownnesses.reshape(-1, 1) # To match up again

            best_qvalue = (knownnesses * best_qvalue) + (
                (1 - knownnesses) * upper_bound)
        
        if novelty_tracker is not None and use_exploration_if_enabled and isinstance(novelty_tracker, DiscretizedStateActionCountingBonus):
            counts = novelty_tracker.get_counts(s.detach().cpu().numpy(), best_actions.detach().cpu().numpy())
            bootstrap_bonuses = self.params["bootstrap_counting_scaling"] * (counts + 1.)**-5
            bootstrap_bonuses = torch.FloatTensor(bootstrap_bonuses).to(self.device).reshape(-1,1)
            best_qvalue = best_qvalue + bootstrap_bonuses
        

        if s.shape[0] == 1:
            a = best_actions[0]
            return best_qvalue, a
        else:
            return best_qvalue, best_actions

    def e_greedy_policy(self,
                        s,
                        episode,
                        train_or_test,
                        use_exploration_if_enabled=True):
        epsilon = 1. / numpy.power(episode,
                                   1. / self.params['policy_parameter'])

        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                q, a = self.get_best_qvalue_and_action(
                    torch.FloatTensor(s_matrix).to(self.device),
                    novelty_tracker=self.novelty_tracker or self.counting_module,
                    use_exploration_if_enabled=use_exploration_if_enabled)
                a = a.cpu().numpy()
            self.train()
            return a

    def gaussian_policy(self, s, episode, train_or_test, use_exploration_if_enabled=True):
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            q, a = self.get_best_qvalue_and_action(
                torch.FloatTensor(s_matrix).to(self.device),
                novelty_tracker=self.novelty_tracker,
                use_exploration_if_enabled=use_exploration_if_enabled)
            a = a.cpu().numpy()
        self.train()
        if train_or_test == 'train':
            noise = numpy.random.normal(loc=0.0,
                                        scale=self.params['noise'],
                                        size=len(a))
            a = a + noise
        return a

    def enact_policy(self, s, episode, train_or_test, policy_type="e_greedy", use_exploration_if_enabled=True):
        assert policy_type in ["e_greedy", "gaussian",], f"Bad policy type: {policy_type}"
        polciy_types = {
            'e_greedy': self.e_greedy_policy,
            'gaussian': self.gaussian_policy,
        }

        return polciy_types[policy_type](s, episode, train_or_test, use_exploration_if_enabled=use_exploration_if_enabled)



    def update(self, target_Q, return_logging_info=False):

        if len(self.buffer_object) < self.params['batch_size']:
            return
        else:
            pass
        s_matrix_np, a_matrix_np, r_matrix_np, done_matrix_np, sp_matrix_np = self.buffer_object.sample(self.params['batch_size'])
        r_matrix_np = numpy.clip(r_matrix_np,
                              a_min=-self.params['reward_clip'],
                              a_max=self.params['reward_clip'])

        s_matrix = torch.FloatTensor(s_matrix_np).to(self.device)
        a_matrix = torch.FloatTensor(a_matrix_np).to(self.device)
        r_matrix = torch.FloatTensor(r_matrix_np).to(self.device)
        sp_matrix = torch.FloatTensor(sp_matrix_np).to(self.device)
        done_matrix = torch.FloatTensor(done_matrix_np).to(self.device)

        with torch.no_grad():
            if self.use_knownness:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=self.novelty_tracker,
                    use_exploration_if_enabled=True,
                    add_action_noise=True)  # this now returns actions
            elif self.use_counting:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=self.counting_module,
                    use_exploration_if_enabled=self.params["bootstrap_counts"])
                # Counts are on the current state, not the future states.
                counts_np = self.counting_module.get_counts(s_matrix_np, a_matrix_np)
                assert counts_np.min() > 0, counts_np.min()
                counts = torch.FloatTensor(counts_np).to(self.device).reshape(-1,1)
                average_unscaled_count_bonus = (counts**-0.5)
                Q_star = self.params["counting_scaling"] * average_unscaled_count_bonus + Q_star
            elif self.use_rnd:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False,
                    add_action_noise=True)
                rnd_bonus = self.rnd_module.get_exploration_bonus_for_states(sp_matrix)
                Q_star = rnd_bonus * self.params['rnd_scaling'] + Q_star
            elif self.use_mpe_bonus:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False,
                    add_action_noise=True)
                mpe_bonus = self.mpe_module.get_exploration_bonus_for_state_actions(s_matrix, a_matrix, sp_matrix)
                Q_star = mpe_bonus * self.params['mpe_scaling'] + Q_star
            else:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False,
                    add_action_noise=True)

        Q_star = Q_star.view((self.params['batch_size'], -1))
        y_q_learning = r_matrix + self.params['gamma'] * (1 -
                                                          done_matrix) * Q_star

        if self.use_exploration:
            novelty_r_matrix = self.novelty_tracker.get_batched_exploration_bonus(
                s_matrix.cpu().numpy())
            novelty_r_matrix = novelty_r_matrix.reshape(
                self.params['batch_size'], 1)
            y_q_learning += novelty_r_matrix


        y = y_q_learning

        y_hat_1, y_hat_2 = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat_1, y) + self.criterion(y_hat_2, y)

        # First, value optimizer.
        self.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self.zero_grad()

        # Then, policy optimizer.
        best_actions = self.policy_module(s_matrix) * self.max_a
        best_qvalue = self.value_module_1(torch.cat([s_matrix, best_actions], dim=1)) # just need to do the one of them!
        policy_loss = -best_qvalue.mean() # make it big!
        policy_loss.backward()
        self.policy_optimizer.step()
        self.zero_grad()

        if self.use_rnd:
            self.rnd_module.update(sp_matrix)
        if self.use_mpe_bonus:
            self.mpe_module.update(s_matrix, a_matrix, sp_matrix)

        utils_for_q_learning.sync_networks(
            target=target_Q,
            online=self,
            alpha=self.params['target_network_learning_rate'],
            copy=False)
        if return_logging_info:
            logging_info = {}
            logging_info['loss'] = loss.cpu().item()
            logging_info['average_q'] = y_hat_1.mean().cpu().item()
            logging_info['average_q_target'] = y.mean().cpu().item()
            return logging_info

    def add_trajectory_to_mmc_augmented_buffer(self, trajectory):
        episodic_return = 0.
        for state, action, reward, next_state, is_terminal in reversed(
                trajectory):
            episodic_return = reward + (self.params["gamma"] * episodic_return)
            mmc_sample_return = deepcopy(episodic_return)
            self.buffer_object.append(state, action, reward, mmc_sample_return,
                                      is_terminal, next_state)

    def add_trajectory_to_vanilla_buffer(self, trajectory):
        for state, action, reward, next_state, is_terminal in trajectory:
            self.buffer_object.append(state, action, reward, is_terminal,
                                      next_state)

    def add_trajectory_to_novelty_tracker(self, trajectory, novelty_tracker):
        states = np.array([s for s, a, r, ns, d in trajectory])
        actions = np.array([a for s, a, r, ns, d in trajectory])
        novelty_tracker.add_many_transitions(states, actions)
        if self.use_exploration:
            self.novelty_tracker.perform_normalization()

    def add_trajectory_to_replay_buffer(self, trajectory):
        self.add_trajectory_to_vanilla_buffer(trajectory)

        if self.use_exploration:
            raise Exception("Not doing this anymore")
        if self.use_knownness:
            self.add_trajectory_to_novelty_tracker(trajectory, self.novelty_tracker)
        if self.use_counting:
            self.add_trajectory_to_novelty_tracker(trajectory, self.counting_module)
