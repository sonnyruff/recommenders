import numpy as np


def normalize_user_ratings(utility_matrix: np.array) -> np.array:
    """
    Normalizes the ratings for all the users
    :param utility_matrix: the utility matrix which has data
    :return:
    """
    return np.linalg.norm(utility_matrix, axis=1)
