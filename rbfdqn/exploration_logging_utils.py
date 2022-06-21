import numpy as np
from rbfdqn.exploration import get_all_distances_to_buffer


def _make_core_set(starting_set, starting_core_set=None, radius=1.):
    # We'll pass in a starting set.
    # starting_set = np.random.random((n, d_dim))
    # Assumes that the initial core set doesn't at all overlap with the starting_set.

    # It's doing it quickly against the current core set, and then whatever
    # remains is guaranteed to not be in the core-set, so we can then only compare against the new stuff.
    if starting_core_set is not None and len(starting_core_set) != 0:
        starting_set = _filter_to_novel_states_with_core_set(
            states_to_add=starting_set,
            starting_core_set=starting_core_set,
            radius=radius)
        core_set = [arr for arr in starting_core_set]  # make into list if not.
    else:
        core_set = []

    comparisons = 0
    while len(starting_set) != 0:
        # Compare to all other elements.
        comparisons += (len(starting_set) - 1)
        first_element = starting_set[0]
        core_set.append(first_element)
        others = starting_set[1:]
        distances = ((others - first_element[None, ...])**2).sum(axis=1)**0.5
        out_of_ball_indices = [
            i for i, d in enumerate(distances) if d >= radius
        ]
        new_starting_set = np.take(others, indices=out_of_ball_indices, axis=0)
        starting_set = new_starting_set

    return core_set



def _filter_to_novel_states_with_core_set(*,
                                          states_to_add,
                                          starting_core_set,
                                          radius=1.,
                                          return_indices=False):
    """
    We filter out all states_to_add that are already covered by something in the coreset.
    """

    np_core_set = np.array(starting_core_set)
    np_states_to_add = np.array(states_to_add)

    distances = get_all_distances_to_buffer(np_states_to_add, np_core_set)
    min_distances = distances.min(axis=1)
    # to make sure I did the right axis.
    assert len(min_distances) == len(states_to_add)
    out_of_ball_indices = [
        i for i, d in enumerate(min_distances) if d >= radius
    ]
    if return_indices:
        return out_of_ball_indices
    else:
        new_states_to_add = np.take(np_states_to_add,
                                    indices=out_of_ball_indices,
                                    axis=0)
        return new_states_to_add


def get_coreset_volume(states,
                       actions=None,
                       starting_core_set=None,
                       radius=1.):
    """
    You pass in stuff from your buffer, and it
    should return something that tracks "volume".
    """

    if actions is not None:
        # Concat them...
        states = np.concatenate((states, actions), axis=1)

    core_sets = _make_core_set(states,
                               starting_core_set=starting_core_set,
                               radius=radius)

    return core_sets


class CoreSetCreator:
    """
    This one is a bit different -- we keep track of the original guys, and do the normalization internally.
    It's for the knownness calculators.
    """
    def __init__(self, radius=1., normalizer=1., track_all_states=False):
        """
        Args:
            radius (float, optional): [description]. Radius that you filter over (after scaling)
            normalizer (np.ndarray, optional): How you scale the data for comparison
            track_all_states (bool, optional): [description]. Whether you keep around all the states. Doing so could be useful for logging maybe.
        """
        self.radius = radius
        self.normalizer = normalizer  #to_divide.
        self.coreset = None
        self.normalized_coreset = None

    def set_normalizer(self, normalizer):
        # WE ONLY WANT TO DO THIS FOR ADAPTIVE GUYS.
        self.normalizer = normalizer
        self._remake_normalized_coreset()

    def _remake_normalized_coreset(self):
        if self.coreset is not None:
            self.normalized_coreset = self.coreset / self.normalizer

    def reset(self):
        self.coreset = None
        self.normalized_coreset = None

    def _filter_to_novel_states_with_core_set(self, states):
        """
        Returns the states that don't contradict with the current coreset.
        Returns the original states, does calculation based on un-normalized.
        """
        states = np.array(states)
        if self.normalized_coreset is None or len(
                self.normalized_coreset) == 0:
            return states

        normalized_states = states / self.normalizer
        distances = get_all_distances_to_buffer(normalized_states,
                                                self.normalized_coreset)
        min_distances = distances.min(axis=1)
        assert len(min_distances) == len(normalized_states)
        out_of_ball_indices = [
            i for i, d in enumerate(min_distances) if d > self.radius
        ]
        states_to_maybe_add = np.take(states,
                                      indices=out_of_ball_indices,
                                      axis=0)
        return states_to_maybe_add  # unnormalized

    def add_states(self, states):
        # For each new guy, we compare against all the old guys, and then eventually add them in at the end.
        # We can actually do the filtering first, that's super fine.

        states_to_maybe_add = self._filter_to_novel_states_with_core_set(
            states)
        normalized_states_to_maybe_add = states_to_maybe_add / self.normalizer

        # Now, these states do NOT overlap with the current coreset.
        # So, they only need to compare with the other states_to_maybe_add
        new_coreset_states = []
        normalized_new_coreset_states = []

        while len(normalized_states_to_maybe_add) != 0:
            first_element = states_to_maybe_add[0]
            others = states_to_maybe_add[1:]

            normalized_first_element = normalized_states_to_maybe_add[0]
            normalized_others = normalized_states_to_maybe_add[1:]

            new_coreset_states.append(first_element)  # add on the first guy
            normalized_new_coreset_states.append(normalized_first_element)

            # Now filter down the remainder.
            normalized_distances = (
                (normalized_others -
                 normalized_first_element[None, ...])**2).sum(axis=1)**0.5
            out_of_ball_indices = [
                i for i, d in enumerate(normalized_distances)
                if d > self.radius
            ]
            states_to_maybe_add = np.take(others,
                                          indices=out_of_ball_indices,
                                          axis=0)
            normalized_states_to_maybe_add = np.take(
                normalized_others, indices=out_of_ball_indices, axis=0)

        print(f"Adding {len(new_coreset_states)} out of {len(states)} states")

        if self.coreset is None or len(self.coreset) == 0:
            self.coreset = np.array(new_coreset_states)
            self.normalized_coreset = np.array(normalized_new_coreset_states)
        else:
            assert len(new_coreset_states) == len(
                normalized_new_coreset_states)
            if len(new_coreset_states) != 0:
                self.coreset = np.concatenate(
                    [self.coreset, np.array(new_coreset_states)])
                self.normalized_coreset = np.concatenate([
                    self.normalized_coreset,
                    np.array(normalized_new_coreset_states)
                ])

        assert len(self.coreset) == len(self.normalized_coreset)
        print(f"Now there are {len(self.coreset)} coreset states")

    def get_unnormalized_coreset(self):
        return np.copy(self.coreset)
