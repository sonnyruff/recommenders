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
    # avgs = np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1)
    # nonzero_mask = np.where(utility_matrix > 0, 1, 0)
    # return t(t(utility_matrix) - avgs) * nonzero_mask

    avgs = np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1)
    zero_mask = np.where(utility_matrix > 0, 0, 1)
    return t(t(utility_matrix) + avgs * t(zero_mask))


def cosine_similarity(utility_matrix, i, j):
    return np.dot(utility_matrix[i][:], utility_matrix[j][:].T) / (
                np.linalg.norm(utility_matrix[i]) * np.linalg.norm(utility_matrix[j]))


def similarity_matrix(utility_matrix):
    sim_matrix = np.zeros((utility_matrix.shape[0], utility_matrix.shape[0]))
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[0]):
            sim_matrix[i][j] = cosine_similarity(utility_matrix, i, j)
    return sim_matrix
