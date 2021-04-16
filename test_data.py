from recommender_utility import normalize_user_ratings
import numpy as np

test_ratings = np.array([
        [4, 0, 0, 5, 1, 0, 0],
        [5, 5, 4, 0, 0, 0, 0],
        [0, 0, 0, 2, 4, 5, 0],
        [0, 3, 0, 0, 0, 0, 3]
    ])

if __name__ == '__main__':
    print(normalize_user_ratings(test_ratings))