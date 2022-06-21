import gym, sys
import numpy, random

from numpy.lib.financial import _ipmt_dispatcher
from rbfdqn import exploration
from rbfdqn.replay_buffer import buffer_class
from rbfdqn import utils_for_q_learning
from rbfdqn.exploration import (
    OnlyStateExplorationClass, StateKnownnessFromMthNeighbor,
    StateActionKnownness, TorchStateActionKnownness,
    TorchNaiveStateKnownness, StateKnownnessFromMthNeighbor,
    TorchStateActionApproxKnownness, StateActionApproxKnownness,
    RNDExploration, MPEExploration, DiscretizedStateActionCountingBonus)

from rbfdqn.shaping_functions import ShapingFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

EXPL_M = 1


def rbf_function_on_action(centroid_locations, action, beta):
    '''
	centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
	action_set: Tensor [batch x a_dim (action_size)]
	beta: float
		- Parameter for RBF function

	Description: Computes the RBF function given centroid_locations and one action
	'''
    assert len(centroid_locations.shape
               ) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(
        action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(
        dim=1).expand_as(centroid_locations)
    diff_norm = diff_norm**2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm +
                           1e-5)
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights


def rbf_function(centroid_locations, action_set, beta):
    '''
	centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
	action_set: Tensor [batch x num_act x a_dim (action_size)]
		- Note: pass in num_act = 1 if you want a single action evaluated
	beta: float
		- Parameter for RBF function

	Description: Computes the RBF function given centroid_locations and some actions
	'''
    assert len(centroid_locations.shape
               ) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape
               ) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"
    diff_norm = torch.cdist(centroid_locations, action_set,
                            p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights


class RandomAgent:
    def __init__(self, params, env, state_size, action_size, **kwargs):
        self.action_size = action_size
        self.env = env
        self.params = params
        self.buffer_object = buffer_class.buffer_class(
            max_length=self.params['max_buffer_size'], env=self.env)
        self.state_size = state_size

    def add_trajectory_to_replay_buffer(self, trajectory):
        for state, action, reward, next_state, is_terminal in trajectory:
            self.buffer_object.append(state, action, reward, is_terminal,
                                      next_state)

    def enact_policy(self, *args, **kwargs):
        a = self.env.action_space.sample()
        return a.tolist()

    def e_greedy_policy(self, *args, **kwargs):
        a = self.env.action_space.sample()
        return a.tolist()

    def update(self, *args, **kwargs):
        return

    def eval(self):
        return


class Net(nn.Module):
    def __init__(self,
                 params,
                 env,
                 state_size,
                 action_size,
                 use_exploration=False,
                 use_counting=False,
                 use_rnd=False,
                 use_mpe_bonus=False,
                 use_knownness=False,
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
        super(Net, self).__init__()

        self.device = device

        self.env = env
        self.params = params
        self.N = self.params['num_points']
        assert len(
            set(self.env.action_space.high[0].reshape(-1))
        )  # Make sure all the actions are the same scale, otherwise bug below.
        self.max_a = torch.FloatTensor(self.env.action_space.high).to(
            self.device)
        self.beta = self.params['temperature']
        num_exploration_modules_loaded = len(list(filter(None, [use_exploration, use_counting, use_knownness, use_rnd, use_mpe_bonus])))
        assert num_exploration_modules_loaded <= 1, f"Can't load more than one exploration module, loaded {num_exploration_modules_loaded}"
        self.use_exploration = use_exploration
        self.use_counting = use_counting
        self.use_rnd = use_rnd
        self.use_mpe_bonus = use_mpe_bonus
        self.use_knownness = use_knownness
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
        self.counting_module = None  # In case we access it at some point.

        if self.use_knownness:
            action_scaling = params.get('action_scaling', 1.0)
            if self.use_torch_knownness:
                if self.only_state_knownness:
                    self.novelty_tracker = TorchNaiveStateKnownness(
                        m=EXPL_M,
                        epsilon=self.params['counting_epsilon'],
                        # normalize=True,
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
                        mapping_type=knownness_mapping_type,
                        normalize=self.normalize_exploration,
                    )
                    # knownness_scaling_array=knownness_scaling_array)
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

        self.value_module = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.N),
        )

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
            torch.nn.init.xavier_uniform_(self.location_module[0].weight)
            torch.nn.init.zeros_(self.location_module[0].bias)
            self.location_module[3].weight.data.uniform_(-.1, .1)
            self.location_module[3].bias.data.uniform_(-1., 1.)

        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size'],
                          self.params['layer_size']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
            torch.nn.init.xavier_uniform_(self.location_module[0].weight)
            torch.nn.init.zeros_(self.location_module[0].bias)
            torch.nn.init.xavier_uniform_(self.location_module[3].weight)
            torch.nn.init.zeros_(self.location_module[3].bias)
            self.location_module[6].weight.data.uniform_(-.1, .1)
            self.location_module[6].bias.data.uniform_(-1., 1.)

        self.criterion = nn.MSELoss()

        # Warning: needs to happen BEFORE optimizer
        # https://pytorch.org/docs/stable/optim.html#constructing-it
        print("Moving to device BEFORE we do the optimizer. ")
        self.to(self.device)

        self.params_dic = [{
            'params': self.value_module.parameters(),
            'lr': self.params['learning_rate']
        }, {
            'params': self.location_module.parameters(),
            'lr': self.params['learning_rate_location_side']
        }]
        try:
            if self.params['optimizer'] == 'RMSprop':
                self.optimizer = optim.RMSprop(self.params_dic)
            elif self.params['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.params_dic)
            else:
                print('unknown optimizer ....')
        except:
            print("no optimizer specified ... ")


    def forward(self, s, a):
        centroid_values = self.get_centroid_values(s)
        centroid_locations = self.get_centroid_locations(s)
        centroid_weights = rbf_function_on_action(centroid_locations, a,
                                                  self.beta)
        output = torch.mul(centroid_weights, centroid_values)
        output = output.sum(1, keepdim=True)
        return output


    def get_centroid_values(self, s):
        '''
        given a batch of s, get all centroid values, [batch x N]
        '''
        centroid_values = self.value_module(s)
        if self.use_naive_optimism:
            centroid_values += self.q_max
        return centroid_values

    def get_centroid_locations(self, s):
        '''
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        '''
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_centroid(self,
                          s,
                          maxOrmin='max',
                          use_exploration_if_enabled=True):
        """
        This returns a value and an action, not a value and an index.

        Singular state!
        """
        all_centroids = self.get_centroid_locations(s)
        weights = rbf_function_single(all_centroids, self.beta, self.N,
                                      self.params['norm_smoothing'])
        values = self.get_centroid_values(s)

        values = torch.transpose(values, 0, 1)
        temp = torch.mm(weights, values)
        temp = temp.detach()

        if use_exploration_if_enabled and self.use_counting:
            print("Not doing this for action selection just yet.")
            pass

        if use_exploration_if_enabled and self.use_knownness:
            np_centroids = torch.cat(all_centroids, dim=0).detach().numpy()
            np_centroids = np.expand_dims(np_centroids, axis=0)

            knownness = self.novelty_tracker.get_knownness_multiple_actions(
                s, np_centroids)
            knownness = torch.FloatTensor(knownness)
            knownness = torch.transpose(
                knownness, 0,
                1)  # NOTE THAT np.transpose works way differently!!!


            assert temp.shape == knownness.shape, "temp shape: {} != knownness shape: {}".format(
                temp.shape, knownness.shape)

            if self.use_shaping:
                # A single number!
                upper_bound = self.shaping_class.get_values(s).item()
            else:
                upper_bound = self.q_max

            # Note the minus here isn't a mistake, it's (knownness - 1), so it's reversing it. For pytorchy reasons.
            temp = (temp * knownness) - (knownness - 1) * upper_bound
        if maxOrmin == 'max':
            values, indices = temp.max(0)
        elif maxOrmin == 'min':
            values, indices = temp.min(0)
        Q_star = values.data.numpy()[0]

        index_star = indices.data.numpy()[0]
        a_star = list(all_centroids[index_star].data.numpy()[0])
        return Q_star, a_star

    def get_best_qvalue_and_action(self,
                                   s,
                                   novelty_tracker=None,
                                   use_exploration_if_enabled=True,
                                   return_batch_action=False):
        '''
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        return_batch_action is False by default because it may be slow and its only used for logging
        '''
        all_centroids = self.get_centroid_locations(s)
        values = self.get_centroid_values(s)
        weights = rbf_function(all_centroids, all_centroids,
                               self.beta)  # [batch x N x N]
        allQ = torch.bmm(weights,
                         values.unsqueeze(2)).squeeze(2)  # bs x num_centroids

        if self.use_shaping:
            upper_bound = self.shaping_class.get_values(s)
            upper_bound = upper_bound.view(-1, 1)
        else:
            upper_bound = self.q_max

        if self.clip_q_targets:
            if isinstance(upper_bound, torch.Tensor):
                allQ = torch.min(allQ, upper_bound)
            elif isinstance(upper_bound, np.ndarray):
                allQ = torch.min(allQ, torch.Tensor(upper_bound))
            elif isinstance(upper_bound, (float, int)):
                allQ = torch.clamp(allQ, max=upper_bound)


        if novelty_tracker is not None and use_exploration_if_enabled and not isinstance(novelty_tracker, DiscretizedStateActionCountingBonus):
            np_centroids = all_centroids.cpu().numpy()
            knownnesses = novelty_tracker.get_knownness_multiple_actions(
                s.detach().cpu().numpy(), np_centroids)
            knownnesses = torch.FloatTensor(knownnesses).to(self.device)

            transformed_knownnesses = knownnesses

            allQ = (transformed_knownnesses * allQ) + (
                (1 - transformed_knownnesses) * upper_bound)

        if novelty_tracker is not None and use_exploration_if_enabled and isinstance(novelty_tracker, DiscretizedStateActionCountingBonus):
            np_centroids = all_centroids.cpu().numpy()
            counts = novelty_tracker.get_count_multiple_actions(s.detach().cpu().numpy(), np_centroids)
            bootstrap_bonuses = self.params["bootstrap_counting_scaling"] * (counts + 1.)**-5
            bootstrap_bonuses = torch.FloatTensor(bootstrap_bonuses).to(self.device)
            allQ = allQ + bootstrap_bonuses

        best, indices = allQ.max(dim=1)

        if s.shape[0] == 1:
            index_star = indices.item()
            a = all_centroids[0, index_star]
            return best, a
        else:
            if return_batch_action: # for speed, only do it if asked.
                best_actions = all_centroids[np.arange(all_centroids.shape[0]), indices, :]
                return best, best_actions
            else:
                return best, None

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
        policy_types = {
            'e_greedy': self.e_greedy_policy,
            'gaussian': self.gaussian_policy,
        }

        return policy_types[policy_type](s, episode, train_or_test, use_exploration_if_enabled=use_exploration_if_enabled)



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
                    use_exploration_if_enabled=True)  # this now returns actions
            elif self.use_counting:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=self.counting_module,
                    use_exploration_if_enabled=self.params["bootstrap_counts"])
                # Counts are on the current state, not the future states.
                counts_np = self.counting_module.get_counts(s_matrix_np, a_matrix_np)
                assert counts_np.min() > 0, counts_np.min()
                counts = torch.FloatTensor(counts_np).to(self.device)
                average_unscaled_count_bonus = (counts**-0.5)
                Q_star = self.params["counting_scaling"] * average_unscaled_count_bonus + Q_star

            elif self.use_rnd:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False)
                rnd_bonus = self.rnd_module.get_exploration_bonus_for_states(sp_matrix)
                Q_star = rnd_bonus * self.params['rnd_scaling'] + Q_star
            elif self.use_mpe_bonus:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False)
                mpe_bonus = self.mpe_module.get_exploration_bonus_for_state_actions(s_matrix, a_matrix, sp_matrix)
                Q_star = mpe_bonus * self.params['mpe_scaling'] + Q_star
            else:
                Q_star, _ = target_Q.get_best_qvalue_and_action(
                    sp_matrix,
                    novelty_tracker=None,
                    use_exploration_if_enabled=False)

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

        y_hat = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat, y)

        self.zero_grad()
        loss.backward()
        self.optimizer.step()
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
            logging_info['average_q'] = y_hat.mean().cpu().item()
            logging_info['average_q_target'] = y.mean().cpu().item()
            if self.use_counting:
                logging_info['average_unscaled_count_bonus'] = average_unscaled_count_bonus.mean().cpu().item()
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
            raise Exception("We don't do this anymore")
        if self.use_knownness:
            self.add_trajectory_to_novelty_tracker(trajectory, self.novelty_tracker)
        if self.use_counting:
            self.add_trajectory_to_novelty_tracker(trajectory, self.counting_module)
