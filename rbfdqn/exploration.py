from collections import defaultdict
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.neighbors import BallTree

import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F

from rbfdqn.normalizers import RunningMeanStd

"""
Counting and Knownness exploratoin bonus modules
"""


def torch_get_all_distances_to_buffer(states, buffer):
    return torch.cdist(states, buffer, p=2)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start



def get_all_distances_to_buffer(states, buffer):
    distances = scipy_cdist(states, buffer)
    return distances


class MPEExploration:
    def __init__(self, *, state_dim, action_dim, hidden_dim=128, device='cpu', use_reward_normalization=False, predict_residual=False):
        """
        Model Prediction Error exploration
        args:
            use_reward_normalization: Whether to do the RND-like thing of using reward normalization. Could be good.
            predict_residual: predict the difference from the last state, versus the entire new state.
        """
        self.use_reward_normalization = use_reward_normalization
        self.predict_residual = predict_residual
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.device = device

        self.predictor = self._create_network()

        self.predictor.to(self.device)

        self.predictor.train()

        self.reward_normalizer = RunningMeanStd(shape=(self.state_dim, ), device=self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters())
        print(f"\n\nAm I using normalizer? {use_reward_normalization}.\n\n")



    def _create_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        if self.predict_residual:
            print("\n\nFor now I'm doing zero init for the final layer of the network for the residual predictor.\n\n")
            torch.nn.init.zeros_(network[-1].weight)
            torch.nn.init.zeros_(network[-1].bias)

        return network

    def predict_next_state(self, states, actions):
        sa_concat = torch.cat([states, actions], dim=1)
        output = self.predictor(sa_concat)
        if self.predict_residual:
            output = output + states
        return output


    def update(self, states, actions, next_states):
        """
        Udates states and next states.
        Takes in a batch of states, and updates the network parameters
        We'll do the normalization tracking here.
        """

        predicted_next_state = self.predict_next_state(states, actions)
        mse = F.mse_loss(predicted_next_state, next_states, reduction='none')

        if self.use_reward_normalization:
            with torch.no_grad(): # updates mean and var
                self.reward_normalizer.update(mse)

        loss = mse.mean()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_exploration_bonus_for_state_actions(self, states, actions, next_states):

        with torch.no_grad():
            predicted_next_states  = self.predict_next_state(states, actions)
            mse = F.mse_loss(predicted_next_states, next_states, reduction='none')
            if self.use_reward_normalization:
                mse = self.reward_normalizer.normalize_batch(mse)

            bonus = mse.mean(dim=1)
            assert bonus.shape[0] == states.shape[0]
            assert len(bonus.shape) == 1

        return bonus.detach()


class RNDExploration:
    """
    Start out with no state normalization. Try and mimic the Novelty API as much as possible.
    """
    def __init__(self, *, state_dim, output_dim=2, hidden_dim=128, device='cpu'):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.device = device

        self.target = self._create_network()
        self.predictor = self._create_network()

        self.target.to(self.device)
        self.predictor.to(self.device)

        self.predictor.train()
        self.target.eval()

        self.reward_normalizer = RunningMeanStd(shape=(self.output_dim, ), device=self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters())



    def _create_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        return network

    def update(self, states):
        """
        Takes in a batch of states, and updates the network parameters
        We'll do the noramlization tracking here.
        """

        with torch.no_grad():
            target_output = self.target(states)
        predictor_output = self.predictor(states)
        mse = F.mse_loss(target_output, predictor_output, reduction='none')

        with torch.no_grad(): # updates mean and var
            self.reward_normalizer.update(mse)

        normalized_mse = self.reward_normalizer.normalize_batch(mse)

        loss = normalized_mse.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_exploration_bonus_for_states(self, states):
        with torch.no_grad():
            target_output = self.target(states)
            predictor_output = self.predictor(states)
            mse = F.mse_loss(target_output, predictor_output, reduction='none')
            normalized_mse = self.reward_normalizer.normalize_batch(mse)

            bonus = normalized_mse.mean(dim=1)
            assert bonus.shape[0] == states.shape[0]
            assert len(bonus.shape) == 1

        return bonus.detach()


class BaseNoveltyMixin:
    def add_many_transitions(self, states, actions=None):
        self.state_dim = states.shape[-1]
        if actions is not None:
            self.action_dim = actions.shape[-1]
        else:
            self.action_dim = None

        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)

        if len(self.unnormalized_buffer) == 0:
            self.unnormalized_buffer = states
        else:
            self.unnormalized_buffer = np.concatenate(
                (self.unnormalized_buffer, states), axis=0)

        self.perform_normalization()

        self._post_add_transitions()

    def perform_normalization(self):

        knownness_scaling_array = getattr(self, 'knownness_scaling_array', 1.)
        if knownness_scaling_array is None:
            knownness_scaling_array = 1.

        knownness_scaling_array = np.maximum(
            knownness_scaling_array, 1e-10)  # avoids divide by 0 errors.
        if not self.normalize:
            self.to_subtract = 0.
            self.to_divide = 1 / knownness_scaling_array
        else:
            min_buffer_elem = self.unnormalized_buffer.min(axis=0)
            max_buffer_elem = self.unnormalized_buffer.max(axis=0)

            if getattr(self, "safen_bounds", False):
                self.to_subtract = min_buffer_elem - 1e-7 # Makes sure the final element is above
                self.to_divide = ((max_buffer_elem - min_buffer_elem) / knownness_scaling_array) + 1e-6
            else:
                self.to_subtract = min_buffer_elem
                self.to_divide = ((max_buffer_elem - min_buffer_elem) /
                                  knownness_scaling_array) + 1e-6

    def normalize_states(self, states):
        return (states - self.to_subtract) / self.to_divide

    def _post_add_transitions(self):
        raise NotImplementedError(
            "This is where you do stuff like make your KDTree.")

    def _num_epsilons_to_knownness(self, num_epsilons_away):
        """
        Converts num_epsilons_away to knownness. Standard will be negative exponential, but we'll do other stuff too.
        """
        mapping_type = getattr(self, 'mapping_type', 'exponential')
        is_torch = isinstance(num_epsilons_away, torch.Tensor)
        if mapping_type == 'exponential':
            to_return = torch.exp(-num_epsilons_away) if is_torch else np.exp(
                -num_epsilons_away)
        elif mapping_type == 'normal':
            to_return = torch.exp(
                -(num_epsilons_away**2)) if is_torch else np.exp(
                    -(num_epsilons_away**2))
        elif mapping_type == 'polynomial':
            # This matches the normal kernel near the origin, but falls off much slower.
            to_return = (1 + num_epsilons_away**2)**-1
        elif mapping_type == "sharp_polynomial":
            # It looks like exponential at the origin, but falls off much slower. 
                to_return = (1 + num_epsilons_away)**-2
        elif mapping_type == "one_over_x_plus_one":
            to_return = (1 + num_epsilons_away)**-1 # sharp but falls away even slower.
        elif mapping_type == "hard":
            if is_torch:
                ones = torch.ones_like(num_epsilons_away)
                zeros = torch.zeros_like(num_epsilons_away)
                to_return = torch.where(num_epsilons_away > 1., ones,
                                        zeros)  # far away chooses zeros.
            else:
                ones = np.ones_like(num_epsilons_away)
                zeros = np.zeros_like(num_epsilons_away)
                to_return = np.where(num_epsilons_away > 1., ones,
                                     zeros)  # far away chooses zeros.

        else:
            raise ValueError(f"Bad argument for knownness_mapping_type. Got {mapping_type}")
        return to_return


class ApproxNoveltyMixin(BaseNoveltyMixin):

    def add_many_transitions(self, states, actions=None):
        if self.normalize:
            return self._add_many_transitions_with_normalization(states, actions=actions)
        else:
            return self._add_many_transitions_without_normalization(states, actions=actions)

    def _add_many_transitions_with_normalization(self, states, actions=None):
        """
        This will manage the filtering, update the normalizer, and change the normalized buffer, etc.
        """
        # Set it up!
        print('doing normalization stuff')
        if not getattr(self, 'coreset_creator', None):
            assert self.normalize == True, "Don't use this for non-adaptive, its super slow"
            assert self.m == 1, "Only works for single states for now!"
            from rbfdqn.exploration_logging_utils import CoreSetCreator # to avoid circular imports.
            self.coreset_creator = CoreSetCreator(radius=self.filter_radius, normalizer=1.) # will re-set normalizer later.

        self.state_dim = states.shape[-1]
        if actions is not None:
            self.action_dim = actions.shape[-1]
        else:
            self.action_dim = None

        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)


        if getattr(self, 'full_unnormalized_buffer', None) is None:
            self.full_unnormalized_buffer = states
        else:
            self.full_unnormalized_buffer = np.concatenate(
                (self.full_unnormalized_buffer, states), axis=0)

        # Just a temp thing for normalization.
        self.unnormalized_buffer = self.full_unnormalized_buffer
        self.perform_normalization()

        self.coreset_creator.set_normalizer(self.to_divide)
        self.coreset_creator.add_states(states)


        self.unnormalized_buffer = self.coreset_creator.get_unnormalized_coreset()

        self._post_add_transitions()

    def _add_many_transitions_without_normalization(self, states, actions=None):
        """
        This will manage the filtering. It may need to make arrays for the new states
        """
        # Set it up!
        print('doing non-normalization stuff')
        if not getattr(self, 'coreset_creator', None):
            assert self.normalize == False, "We can't be doing buffer-based normalization here, at least for now."
            assert self.m == 1, "Only works for single states for now!"
            self.perform_normalization() # should never change! Because normalize is false.
            from rbfdqn.exploration_logging_utils import CoreSetCreator # to avoid circular imports.
            self.coreset_creator = CoreSetCreator(radius=self.filter_radius, normalizer=self.to_divide)

        # if states is not None:
        self.state_dim = states.shape[-1]
        if actions is not None:
            self.action_dim = actions.shape[-1]
        else:
            self.action_dim = None

        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)

        self.coreset_creator.add_states(states)

        self.unnormalized_buffer = self.coreset_creator.get_unnormalized_coreset()

        self._post_add_transitions()
    
    def _num_epsilons_to_knownness(self, num_epsilons_away):
        """
        This is the same as the other one, except we need to take into account that there
        may have been a closer guy that got filtered away. So, we just subtract "radius/epsilon".
        """
        num_epsilons_in_radius = (self.filter_radius/self.epsilon)
        num_epsilons_away = num_epsilons_away - num_epsilons_in_radius

        # Can't have negatives.
        is_torch = isinstance(num_epsilons_away, torch.Tensor)
        num_epsilons_away = num_epsilons_away.clamp(0.) if is_torch else num_epsilons_away.clip(0.)
        return super()._num_epsilons_to_knownness(num_epsilons_away)



class OnlyStateMixin:
    def get_knownness_multiple_actions(self, states, actions):
        # Pretty much a dummy method to make sure the APIs are the same..
        assert len(states) == len(actions), "{} != {}".format(
            len(states), len(actions))
        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 3, actions.shape

        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        num_actions_per_state = actions.shape[1]
        knownnesses = self.get_knownness(states)
        assert len(knownnesses.shape) == 1
        knownnesses = knownnesses.reshape(-1, 1)
        knownnesses = np.repeat(knownnesses, num_actions_per_state, axis=1)
        assert knownnesses.shape == (len(states), len(actions[0]))
        return knownnesses


class StateActionMixin:
    """
    There's going to be a few different underscored methods. And then, the non-action classes will
    say "if _merge_state_actions" or something like that. That avoids the "super" problem.
    """
    def _concat_states_actions(self, states, actions):
        assert len(states) == len(
            actions), "len(states) is {}, len(actions) is {}".format(
                len(states), len(actions))
        assert len(states.shape) == len(
            actions.shape
        ) == 2, "states.shape is {}, actions.shape is {}".format(
            states.shape, actions.shape)
        sa_concat = np.concatenate((states, actions), axis=1)
        return sa_concat

    def get_knownness_multiple_actions(self, states, actions):
        """
        For example, 10 states, actions will be shape [10, num_centroids, action_dim]
        And states will be [10, state_dim]

        Returns: Something like a bunch of numbers between 0 and 1....
        """
        assert len(states) == len(actions), "{} != {}".format(
            len(states), len(actions))
        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 3, actions.shape

        states_expanded = np.expand_dims(states, axis=1)
        states_filled = np.repeat(states_expanded,
                                  repeats=actions.shape[1],
                                  axis=1)

        states_reshaped = states_filled.reshape((-1, states.shape[-1]))
        actions_reshaped = actions.reshape(-1, actions.shape[-1])

        knownness = self.get_knownness(states_reshaped, actions_reshaped)

        assert knownness.min() >= 0., knownness.min()
        assert knownness.max() <= 1., knownness.max()

        knownness_reshaped = knownness.reshape((actions.shape[0:2]))
        return knownness_reshaped

    def get_count_multiple_actions(self, states, actions):
        """
        For example, 10 states, actions will be shape [10, num_centroids, action_dim]
        And states will be [10, state_dim]
        So, the thing I end up with needs to be [10, num_centroids]
        So, I need to add a dimension to states, and then... tile?

        Returns: Something like a bunch of numbers between 0 and 1....
        """
        assert len(states) == len(actions), "{} != {}".format(
            len(states), len(actions))
        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 3, actions.shape

        # Make the state array match on two dimensions, then do
        # np.dstack (third dimension). Then, reshape and get our answers, and finally\
        # reshape again.
        states_expanded = np.expand_dims(states, axis=1)
        states_filled = np.repeat(states_expanded,
                                  repeats=actions.shape[1],
                                  axis=1)

        states_reshaped = states_filled.reshape((-1, states.shape[-1]))
        actions_reshaped = actions.reshape(-1, actions.shape[-1])

        counts = self.get_counts(states_reshaped, actions_reshaped)

        assert counts.min() >= 0., counts.min() 

        counts_reshaped = counts.reshape((actions.shape[0:2]))
        return counts_reshaped


class TorchKnownnessMixin:
    def __init__(self,
                 *,
                 m,
                 epsilon,
                 normalize=True,
                 batch_size=100,
                 action_scaling=None,
                 mapping_type="exponential",
                 knownness_scaling_array=None,
                 filter_radius=0.):


        if not torch.cuda.is_available():
            print(
                "Are you sure you want to be using the naive method without cuda?!"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} for knownness")

        self.batch_size = batch_size

        self.m = m
        self.epsilon = epsilon
        self.normalize = normalize
        self.mean = 0
        self.std = 1

        self.unnormalized_buffer = np.array([])
        self.normalized_buffer = np.array([])

        self.normalized_buffer_torch = None
        self.action_scaling = action_scaling
        self.mapping_type = mapping_type
        self.knownness_scaling_array = knownness_scaling_array
        self.filter_radius = filter_radius

    def _post_add_transitions(self):
        normalized_buffer = (self.unnormalized_buffer -
                             self.to_subtract) / self.to_divide

        del self.normalized_buffer_torch  # force deallocation.
        torch.cuda.empty_cache()
        self.normalized_buffer_torch = torch.FloatTensor(normalized_buffer).to(
            self.device)

    def get_knownness(self, states, actions=None):
        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)

        if len(self.unnormalized_buffer) < 10:
            return np.array([0 for s in states])

        states = self.normalize_states(states)

        states_torch = torch.FloatTensor(states).to(self.device)

        num_chunks = int(
            np.ceil(len(self.normalized_buffer_torch) / self.batch_size))
        buffer_chunks = [
            self.normalized_buffer_torch[self.batch_size * c:self.batch_size *
                                         (c + 1)] for c in range(num_chunks)
        ]

        # Memory is still a problem. Luckily, we can get around it by doing a topk on each batch, appending,
        # and doing a topk at the end. Point being, you don't need to save anything besides those.
        # That should narrow it down a LOT.
        topk_candidates = []
        for chunk in buffer_chunks:
            distances = torch_get_all_distances_to_buffer(
                states_torch, chunk)
            assert distances.shape[0] == states_torch.shape[0], (
                distances.shape, states_torch.shape)  # TODO: comment out
            assert distances.shape[1] == chunk.shape[0], (
                distances.shape, chunk.shape)  # TODO: comment out

            m_or_batch_size = min(self.m, int(
                distances.shape[1]))  # so it doesn't error on us later.
            topk_batch = torch.topk(distances, m_or_batch_size,
                                    largest=False)[0]
            assert topk_batch.shape[0] == states_torch.shape[0], (
                topk_batch.shape, states_torch.shape)
            assert topk_batch.shape[1] == m_or_batch_size, (topk_batch.shape,
                                                            m_or_batch_size)

            topk_candidates.append(topk_batch)

        all_topk_candidates = torch.cat(topk_candidates, dim=1)
        topk_distances = torch.topk(all_topk_candidates, self.m,
                                    largest=False)[0]

        biggest_distance = topk_distances.max(dim=-1)[0]
        num_epsilons_away = biggest_distance / self.epsilon

        knownness_gpu = self._num_epsilons_to_knownness(num_epsilons_away)

        knownness = knownness_gpu.cpu().numpy()

        # collect cuda garbage
        del distances
        del buffer_chunks
        del topk_batch
        del all_topk_candidates
        del topk_distances
        torch.cuda.empty_cache()

        return knownness

class DiscretizationCountingMixin:
    def __init__(self,
                 *,
                 m,
                 epsilon,
                 normalize=True,
                 action_scaling=None,
                 knownness_scaling_array=None,
                 filter_radius=0.):
        self.epsilon = epsilon
        self.normalize = normalize
        self.action_scaling = action_scaling
        self.unnormalized_buffer = np.array([])
        self.normalized_buffer = np.array([])
        self.knownness_scaling_array = knownness_scaling_array
        self.safen_bounds = True

    def discretize(self, elem, epsilon):
        return tuple((elem / epsilon).floor().astype(int))

    def discretize_batch(self, elems, epsilon):
        """Turns it into a list of tuples"""
        elems_discrete = np.floor(elems / epsilon).astype(int)
        return [tuple(elem) for elem in elems_discrete]

    def _post_add_transitions(self):
        normalized_buffer = (self.unnormalized_buffer -
                             self.to_subtract) / self.to_divide
        # Now, we make the dictionary I guess.
        count_dict = defaultdict(int)
        assert len(normalized_buffer.shape) == 2

        discretized_buffer = self.discretize_batch(normalized_buffer, self.epsilon)
        for discrete_sa in discretized_buffer:
            count_dict[discrete_sa] += 1

        self.count_dict = count_dict

    def get_knownness(self, states, actions=None):
        raise Exception("Ain't know knownness here")    

    def get_counts(self, states, actions=None):
        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)


        if len(self.unnormalized_buffer) < 10:
            return np.array([0 for s in states])

        states = self.normalize_states(states)
        assert len(states.shape) == 2
        discretized_states = self.discretize_batch(states, self.epsilon)
        counts = [self.count_dict[ds] for ds in discretized_states]

        return np.asarray(counts, dtype=np.float32) + 1e-2 # The small addition shouldn't make a difference, when count is 1 it's so tiny

class StandardKnownnessMixin:
    def __init__(self,
                 *,
                 m,
                 epsilon,
                 normalize=True,
                 action_scaling=None,
                 mapping_type='exponential',
                 knownness_scaling_array=None,
                 filter_radius=0.):
        self.m = m
        self.epsilon = epsilon
        self.normalize = normalize
        self.mean = 0
        self.std = 1

        self.unnormalized_buffer = np.array([])
        self.normalized_buffer = np.array([])

        self.tree = None
        self.action_scaling = action_scaling
        self.mapping_type = mapping_type
        self.knownness_scaling_array = knownness_scaling_array
        self.filter_radius = filter_radius

    def _post_add_transitions(self):
        normalized_buffer = self.normalize_states(self.unnormalized_buffer)
        self.tree = BallTree(normalized_buffer)

    def get_knownness(self, states, actions=None):
        if hasattr(self, "_concat_states_actions"):
            states = self._concat_states_actions(states, actions)

        if len(self.unnormalized_buffer) < 10:
            return np.array([0 for s in states])

        states = self.normalize_states(states)

        distances, indices = self.tree.query(states, k=self.m, dualtree=True)
        max_distances = distances.max(axis=1)

        num_epsilons_away = max_distances / self.epsilon
        knownness = self._num_epsilons_to_knownness(num_epsilons_away)
        return knownness


class TorchNaiveStateKnownness(TorchKnownnessMixin, OnlyStateMixin,
                               BaseNoveltyMixin):
    pass


class TorchStateActionKnownness(TorchKnownnessMixin, StateActionMixin,
                                BaseNoveltyMixin):
    pass


class StateKnownnessFromMthNeighbor(StandardKnownnessMixin, OnlyStateMixin,
                                    BaseNoveltyMixin):
    pass


class StateActionKnownness(StandardKnownnessMixin, StateActionMixin,
                           BaseNoveltyMixin):
    pass




class TorchStateActionApproxKnownness(TorchKnownnessMixin, StateActionMixin,
                                ApproxNoveltyMixin):
    pass


class StateActionApproxKnownness(StandardKnownnessMixin, StateActionMixin,
                           ApproxNoveltyMixin):
    pass

class DiscretizedStateActionCountingBonus(DiscretizationCountingMixin, StateActionMixin,
                            BaseNoveltyMixin):
    pass


class OnlyStateExplorationClass:
    def __init__(self, *, epsilon, normalize=True, scaling=1.):
        raise Exception("Decommisioned for now!")
        self.epsilon = epsilon
        self.normalize = normalize
        self.mean = 0
        self.std = 1
        self.scaling = scaling

        self.unnormalized_buffer = np.array([])
        self.normalized_buffer = np.array([])

    def add_transition(self, s, a=None):
        normalized = (s - self.mean) / self.std
        if len(self.unnormalized_buffer) == 0:
            self.unnormalized_buffer = s[None, ...]
            self.normalized_buffer = normalized[None, ...]
        else:
            self.unnormalized_buffer = np.concatenate(
                (self.unnormalized_buffer, s[None, ...]), axis=0)
            self.normalized_buffer = np.concatenate(
                (self.normalized_buffer, normalized[None, ...]), axis=0)

    def add_many_transitions(self, states, actions=None):
        assert len(states.shape) == 2
        normalized_states = (states - self.mean) / self.std
        if len(self.unnormalized_buffer) == 0:
            self.unnormalized_buffer = states
            self.normalized_buffer = normalized_states
        else:
            self.unnormalized_buffer = np.concatenate(
                (self.unnormalized_buffer, states), axis=0)
            self.normalized_buffer = np.concatenate(
                (self.normalized_buffer, normalized_states), axis=0)

    def get_exploration_bonus(self, s, a=None):
        # This should ONLY be used to update based on the current state for now!!!
        if len(self.normalized_buffer) == 0:
            return 0

        normalized = (s - self.mean) / self.std
        difference = self.normalized_buffer - normalized
        distances = (difference**2).sum(axis=1)**0.5

        counts = np.exp(-distances / self.epsilon)

        count_sum = counts.sum()

        exploration_bonus = (count_sum + 1e-6)**-0.5
        return exploration_bonus * self.scaling

    def get_batched_exploration_bonus(self, states, actions=None):
        if len(self.normalized_buffer) == 0:
            return 0

        normalized = (states - self.mean) / self.std

        distances = get_all_distances_to_buffer(normalized,
                                                self.normalized_buffer)

        counts_per_interaction = np.exp(-distances / self.epsilon)
        counts_per_state = counts_per_interaction.sum(axis=1)

        exploration_bonus = (counts_per_state + 1e-6)**-0.5
        exploration_bonus = exploration_bonus * self.scaling

        return exploration_bonus

    def perform_normalization(self):
        self.mean = self.unnormalized_buffer.mean(axis=0)
        self.std = self.unnormalized_buffer.std(axis=0) + 1e-6

        self.normalized_buffer = (self.unnormalized_buffer -
                                  self.mean) / self.std
