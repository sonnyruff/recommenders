import numpy as np
from tqdm import trange

np.set_printoptions(linewidth=150)


def normalize_user_ratings(utility_matrix: np.array) -> np.array:
    """
    Normalizes the ratings for the cosine similarity by subtracting the average per user from all
    the non-zero entries of that user
    :param utility_matrix: the utility matrix which has data
    :return:
    """
    t = np.transpose
    avgs = np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1)
    nonzero_mask = np.where(utility_matrix > 0, 1, 0)
    return t(t(utility_matrix) - avgs) * nonzero_mask

    # avgs = np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1)
    # zero_mask = np.where(utility_matrix > 0, 0, 1)
    # return t(t(utility_matrix) + avgs * t(zero_mask))


def cosine_similarity(utility_matrix, i, j):
    return np.dot(utility_matrix[i][:], utility_matrix[j][:].T) / (
                np.linalg.norm(utility_matrix[i]) * np.linalg.norm(utility_matrix[j]))


def similarity_matrix(utility_matrix):
    sim_matrix = np.zeros((utility_matrix.shape[0], utility_matrix.shape[0]))
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[0]):
            sim_matrix[i][j] = cosine_similarity(utility_matrix, i, j)
    return sim_matrix


def global_baseline(utility_matrix):
    mean_movie_rating = np.sum(utility_matrix) / np.count_nonzero(utility_matrix)
    movie_avgs = np.nan_to_num(np.sum(utility_matrix, axis=0) / np.count_nonzero(utility_matrix, axis=0))
    user_avgs = np.nan_to_num(np.sum(utility_matrix, axis=1) / np.count_nonzero(utility_matrix, axis=1))

    movie_deviation = movie_avgs - np.average(movie_avgs)
    user_deviation = user_avgs - np.average(user_avgs)

    prediction_matrix = np.zeros(utility_matrix.shape)

    for user_id in trange(utility_matrix.shape[0]):
        for movie_id in range(utility_matrix.shape[1]):
            if utility_matrix[user_id][movie_id] == 0:
                prediction_matrix[user_id][movie_id] = mean_movie_rating + movie_deviation[movie_id] + user_deviation[user_id]
            else:
                prediction_matrix[user_id][movie_id] = utility_matrix[user_id][movie_id]

    return prediction_matrix
