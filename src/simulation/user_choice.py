"""
The script to create user utility model and user choice model
"""

import json
import pickle

import numpy as np


def load_json(file_name):
    """
    Load the json file given the file name
    """
    with open(file_name) as f:
        data = json.load(f)

    return data


def load_pickle(file_name):
    """
    Load the pickle file given the file name
    """
    with open(file_name, "rb") as handle:
        pickle_file = pickle.load(handle)

    return pickle_file


def beta_distribution(mu, sigma=10 ** -5):
    """
    Sample from beta distribution given the mean and variance. 
    """
    alpha = mu * mu * ((1 - mu) / (sigma * sigma) - 1 / mu)
    beta = alpha * (1 / mu - 1)

    return np.random.beta(alpha, beta)


def random_select(candidates, size=50):
    """
    Random select a bunch of articles' pks
    """
    llist = list(candidates)

    np.random.shuffle(llist)

    return llist[:size]


def user_interaction(uv, recommended_News, ranked=True):
    """
    Given a user vector (uv) and a recommended new, 
    return whether user is gonna click or not
    """

    iv = recommended_News["topical_vector"]

    product = simple_doct_product(uv, iv)

    epsilon = 10e-5

    if (product + epsilon) > 1.0:
        vui = 0.99
    else:
        vui = beta_distribution(product)

    # Awared preference
    ita = beta_distribution(0.98)
    pui = vui * ita

    rand_num = np.random.random()

    if rand_num < pui:
        return True
    else:
        return False


def user_interaction_score(uv, recommended_News, ranked=True):
    """
    Given a user vector (uv) and a recommended new, 
    return the probability of user's clicking
    """

    iv = recommended_News["topical_vector"]

    product = simple_doct_product(uv, iv)

    epsilon = 10e-5

    if (product + epsilon) > 1.0:
        vui = 0.99
    else:
        vui = beta_distribution(product)

    # Awared preference
    ita = beta_distribution(0.98)
    pui = vui * ita

    return pui


def simple_doct_product(u, v):
    """
    u is user vector, v is item vector
    v should be normalized
    """
    v = [i / (sum(v)) for i in v]

    return np.dot(u, v)
