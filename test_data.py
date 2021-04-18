from recommender_utility import *
import numpy as np
import matplotlib.pyplot as plt

test_ratings = np.array([
        [4, 0, 0, 5, 1, 0, 0],
        [5, 5, 4, 0, 0, 0, 0],
        [0, 0, 0, 2, 4, 5, 0],
        [0, 3, 0, 0, 0, 0, 3]
    ])

if __name__ == '__main__':
    # print(global_baseline(test_ratings))

    user_avgs = np.nan_to_num(np.sum(test_ratings, axis=1) / np.count_nonzero(test_ratings, axis=1))
    print(test_ratings[0, :4].shape)
    # print(np.array(test_ratings[:, 0])[np.newaxis].T)
    print(test_ratings[0, :4] @ test_ratings[:, 0])

    # normalized_user_ratings = normalize_user_ratings(test_ratings)
    # print(cosine_similarity(normalize_user_ratings(test_ratings), 0, 2))
    # print(similarity_matrix(normalized_user_ratings))

    # M = similarity_matrix(normalize_user_ratings(test_ratings))
    # plt.imshow(M)
    # plt.colorbar()
    # plt.show()