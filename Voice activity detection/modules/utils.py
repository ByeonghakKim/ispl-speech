import numpy as np


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def data_transform(inputs, w, u, min_abs_value):

    # """
    # :param inputs. shape = (batch_size, feature_size)
    # :param w : decide neighbors
    # :param u : decide neighbors
    # :return: trans_inputs. shape = (batch_size, feature_size*len(neighbors))
    # """

    neighbors_1 = np.arange(-w, -u, u)
    neighbors_2 = np.array([-1, 0, 1])
    neighbors_3 = np.arange(1+u, w+1, u)

    neighbors = np.concatenate((neighbors_1, neighbors_2, neighbors_3), axis=0)

    pad_size = 2*w + inputs.shape[0]
    pad_inputs = np.ones((pad_size, inputs.shape[1])) * min_abs_value
    pad_inputs[0:inputs.shape[0], :] = inputs

    trans_inputs = [np.roll(pad_inputs, -1*neighbors[i], axis=0)[0:inputs.shape[0], :]
                    for i in range(neighbors.shape[0])]

    trans_inputs = np.asarray(trans_inputs)
    trans_inputs = np.transpose(trans_inputs, [1, 0, 2])

    return trans_inputs


def data_transform_targets(inputs, w, u):

    # """
    # :param inputs. shape = (batch_size, feature_size)
    # :param w : decide neighbors
    # :param u : decide neighbors
    # :return: trans_inputs. shape = (batch_size, feature_size*len(neighbors))
    # """

    neighbors_1 = np.arange(-w, -u, u)
    neighbors_2 = np.array([-1, 0, 1])
    neighbors_3 = np.arange(1+u, w+1, u)

    neighbors = np.concatenate((neighbors_1, neighbors_2, neighbors_3), axis=0)

    pad_size = 2*w + inputs.shape[0]
    pad_inputs = np.zeros(pad_size)
    pad_inputs[0:inputs.shape[0]] = inputs

    trans_inputs = [np.roll(pad_inputs, -1*neighbors[i], axis=0)[0:inputs.shape[0]]
                    for i in range(neighbors.shape[0])]

    trans_inputs = np.asarray(trans_inputs)
    trans_inputs = np.max(trans_inputs, axis=0)

    return trans_inputs


def data_transform_targets_bdnn(inputs, w, u):

    # """
    # :param inputs. shape = (batch_size, feature_size)
    # :param w : decide neighbors
    # :param u : decide neighbors
    # :return: trans_inputs. shape = (batch_size, feature_size*len(neighbors))
    # """

    neighbors_1 = np.arange(-w, -u, u)
    neighbors_2 = np.array([-1, 0, 1])
    neighbors_3 = np.arange(1+u, w+1, u)

    neighbors = np.concatenate((neighbors_1, neighbors_2, neighbors_3), axis=0)

    pad_size = 2*w + inputs.shape[0]
    pad_inputs = np.zeros(pad_size)
    pad_inputs[0:inputs.shape[0]] = inputs

    trans_inputs = [np.roll(pad_inputs, -1*neighbors[i], axis=0)[0:inputs.shape[0]]
                    for i in range(neighbors.shape[0])]

    trans_inputs = np.asarray(trans_inputs)
    trans_inputs = np.transpose(trans_inputs, [1, 0])

    return trans_inputs
