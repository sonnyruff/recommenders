from recommender_utility import normalize_user_ratings, cosine_similarity, similarity_matrix
import numpy as np

test_ratings = np.array([
        [4, 0, 0, 5, 1, 0, 0],
        [5, 5, 4, 0, 0, 0, 0],
        [0, 0, 0, 2, 4, 5, 0],
        [0, 3, 0, 0, 0, 0, 3]
    ])

if __name__ == '__main__':
    normalized_user_ratings = normalize_user_ratings(test_ratings)
    print(cosine_similarity(normalize_user_ratings(test_ratings), 0, 2))
    print(similarity_matrix(normalized_user_ratings))