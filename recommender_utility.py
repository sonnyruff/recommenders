import numpy as np

np.set_printoptions(linewidth=150)


def normalize_user_ratings(utility_matrix: np.array) -> np.array:
    """
    Normalizes the ratings for the cosine similarity by subtracting the average per user from all
    the non-zero entries of that user
    :param utility_matrix: the utility matrix which has data
    :return:
    """
    t = np.transpose
    return t(t(utility_matrix) - (np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1))) \
           * np.where(utility_matrix > 0, 1, 0)
