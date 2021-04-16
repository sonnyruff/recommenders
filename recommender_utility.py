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
    return t(t(utility_matrix) - (np.sum(utility_matrix, axis=1)
                                  / np.count_nonzero(utility_matrix, axis=1))) * np.where(utility_matrix > 0, 1, 0)


def cosine_similarity(utility_matrix, i, j):
    if np.sum(utility_matrix[i]) == 0 or np.sum(utility_matrix[j]) == 0:
        return 0
    return np.dot(utility_matrix[i], utility_matrix[j].T) / (
                np.linalg.norm(utility_matrix[i]) * np.linalg.norm(utility_matrix[j]))


def similarity_matrix(utility_matrix):
    sim_matrix = np.zeros((utility_matrix.shape[0], utility_matrix.shape[0]))
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[0]):
            sim_matrix[i][j] = cosine_similarity(utility_matrix, i, j)
    return sim_matrix
