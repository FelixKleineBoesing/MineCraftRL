import numpy as np
import simplejson
import os
import logging
from collections import OrderedDict


class Config:

    def __init__(self):
        self._load_from_file()

    def _load_from_file(self):
        with open("../../config.json", "r") as f:
            self._store = simplejson.load(f)

    def _get_from_env(self, key: str):
        return os.environ[key]

    def __getitem__(self, item):
        if os.path.isfile("../../is_docker"):
            try:
                return self._get_from_env(item)
            except KeyError:
                logging.error("Key {} is not present in environment variables!")
        else:
            try:
                return self._store[item]
            except KeyError:
                logging.error("Key {} is not present in config.json!")


def multiply(*args):
    """
    Helper function which multiplies the all numbers that are delivered in the function call
    :param args: numbers
    :return: product of all numbers
    """
    product = 1
    for arg in args:
        product *= arg
    return product


def min_max_scaling(arr: np.ndarray, min_val: float=-2.0, max_val: float=2.0):
    """
    scales the given array between 0 and 1
    :param arr: numpy array
    :param min_val: min occurence in data, default is -2.0 based on the min possible stone value
    :param max_val: max occurence in data, default is +2.0 based on the max possible stone value
    :return: scaled numpy array
    """
    return (arr.astype('float32') - min_val) / (max_val - min_val)


def update_managed_dict(managed_dict, game_id, key, value):
    """
    updates a managed dictionary since complex data structures must be called from managed dict, then be updated and
    finally put back into the dictionary

    :param managed_dict: multiprocessing dictionary
    :param game_id:
    :param key:
    :param value:
    :return:
    """
    content = managed_dict[game_id]
    content[key] = value
    managed_dict[game_id] = content


class ActionSpace:

    def __init__(self, action_space):
        """

        :param action_space: action space from gym env
        """
        self.action_space = action_space
        self.number_discrete_actions = 9
        self.length_action_vector = 2 ** 9

    def build_action_vector(self, action_dict: dict):
        actions = ["attack", "back", "forward", "jump", "left", "place", "right", "sneak", "sprint"]
        value = ""
        for action in actions:
            value += str(action_dict[action])
        index = int(value, 2)
        action_vector = np.zeros((self.length_action_vector, 1))
        action_vector[index] = 1

        return action_vector

    def build_action_dict(self, action_vector: np.array):
        actions = ["attack", "back", "forward", "jump", "left", "place", "right", "sneak", "sprint"]
        action_dict = OrderedDict()
        index = np.where(action_vector == 1)[0][0]
        binary_number = bin(index)[2:]
        for action, valu in zip(actions, binary_number):
            action_dict[action] = valu

        return action_dict
