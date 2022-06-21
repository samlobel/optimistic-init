import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="colorblind")
# sns.set()


class MetaLogger:
    """
    Logs the way we like. Maybe should go in the plotting folder because they go together.
    """
    def __init__(self, logging_directory) -> None:
        super().__init__()
        self._logging_directory = logging_directory
        os.makedirs(logging_directory, exist_ok=True)
        self._logging_values = {}
        self._filenames = {}

    def add_field(self, field_name, filename):
        assert isinstance(field_name, str)
        assert field_name != ""
        for char in [" ", "/", "\\"]:
            assert char not in field_name

        folder_name = os.path.join(self._logging_directory, field_name)
        os.makedirs(folder_name, exist_ok=True)
        print(f"Successfully created the directory {folder_name}")

        full_path = os.path.join(folder_name, filename)
        self._filenames[field_name] = full_path

        assert field_name not in self._logging_values

        self._logging_values[field_name] = []

    def append_datapoint(self, field_name, datapoint, write=False):
        self._logging_values[field_name].append(datapoint)
        if write:
            self.write_field(field_name)

    def write_field(self, field_name):
        full_path = self._filenames[field_name]
        values = self._logging_values[field_name]
        with open(full_path, "wb+") as f:
            pickle.dump(values, f)

    def write_all_fields(self):
        for field_name in self._filenames.keys():
            self.write_field(field_name)


def get_scores(log_dir, subdir="scores", only_longest=False, min_length=-1, cumulative=False):
    longest = 0
    print(log_dir)
    overall_scores = []
    glob_pattern = log_dir + "/" + subdir + "/" + "*.pkl"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        if only_longest:
            if len(scores) > longest:
                overall_scores = [scores]
                longest = len(scores)
        else:
            if len(scores) >= min_length:  # -1 always passes!
                overall_scores.append(scores)

    if len(overall_scores) == 0:
        raise Exception("No scores in " + log_dir)

    min_length = min(len(s) for s in overall_scores)

    overall_scores = [s[:min_length] for s in overall_scores]

    score_array = np.array(overall_scores)
    # print(score_array.shape)
    if cumulative:
        score_array = np.cumsum(score_array, axis=1)
    return score_array


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=25):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smoothen_data(scores, n=10):
    print(scores.shape)
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data


def generate_plot(score_array, label, smoothen=False,color=None, linewidth=8):
    if smoothen:
        print('bingbingbing')
        smooth_amount = 15 if isinstance(smoothen, bool) else smoothen
        print(smooth_amount)
        score_array = smoothen_data(score_array, n=smooth_amount)
    median, mean, top, bottom = get_plot_params(score_array)
    # plt.plot(mean, linewidth=2, label=label, alpha=0.9)
    # plt.fill_between(range(len(top)), top, bottom, alpha=0.2)
    if color is not None:
        plt.plot(mean, linewidth=linewidth, label=label, alpha=0.9, c=color)
        plt.fill_between(range(len(top)), top, bottom, alpha=0.35, color=color)
    else:
        plt.plot(mean, linewidth=linewidth, label=label, alpha=0.9)
        plt.fill_between(range(len(top)), top, bottom, alpha=0.35)
