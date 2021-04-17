import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import time
from random import randint
from recommender_utility import *

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)



#####
## COLLABORATIVE FILTERING
#####
def predict_collaborative_filtering(movies, users, ratings, predictions):
    M = construct_util_matrix(movies, users, ratings, predictions)
    # print(ratings.loc[100])

    pass


#####
## LATENT FACTORS
#####
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
## FINAL PREDICTORS
#####
def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
## RANDOM PREDICTORS
#####
# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


####################################################################################################

def normalize_user_ratings(utility_matrix: np.array) -> np.array:
    """
    Normalizes the ratings for all the users
    :param utility_matrix: the utility matrix which has data
    :return:
    """
    return np.linalg.norm(utility_matrix, axis=1)


####################################################################################################


def construct_util_matrix(movies, users, ratings, predictions, limit):
    if limit > 0:
        within_limit = (ratings["userID"]<limit) & (ratings["movieID"]<limit)
        ratings = ratings[within_limit]
        M = np.zeros(shape=(limit, limit))
    else:
        M = np.zeros(shape=(users.shape[0], movies.shape[0]))

    for i, row in ratings.iterrows():
        M[row['userID'] - 1, row['movieID'] - 1] = row['rating']

    return M


####################################################################################################


def compute_column_probabilities(M):
    probabilities = np.zeros(M.shape[1])

    total = np.sum(np.square(M))
    for i in range(M.shape[1]):
        probabilities[i] = np.sum(np.square(M[:, i])) / total

    return probabilities


def compute_row_probabilities(M):
    probabilities = np.zeros(M.shape[0])

    total = np.sum(np.square(M))
    for i in range(M.shape[0]):
        probabilities[i] = np.sum(np.square(M[i, :])) / total

    return probabilities


def construct_c(M, r):
    #     C = np.zeros([M.shape[0], r])

    probs = compute_column_probabilities(M)
    columns = np.random.choice(M.shape[1], r, p=probs)

    C = np.multiply(M[:, columns],
                    np.sqrt(np.multiply(probs[columns], r)))

    return C, columns


def construct_r(M, r):
    probs = compute_row_probabilities(M)
    rows = np.random.choice(M.shape[0], r, p=probs)

    probsT = np.array(probs[rows])[np.newaxis].T

    R = np.multiply(M[rows, :],
                    np.sqrt(np.multiply(probsT, r)))

    return R, rows


def construct_u(M, r, rows, columns):
    #     W = np.zeros([r, r])

    W = M[:, columns][rows, :]

    # compute U from W
    X, E, Yt = np.linalg.svd(W)
    E = np.linalg.pinv(np.diag(E)) ** 2
    U = np.matmul(np.matmul(Yt.T, E), X.T)

    return U


def cur(M, r):
    print("Starting CUR : ", time.strftime('%H:%M:%S.%f'))
    C, columns = construct_c(M, r)
    R, rows = construct_r(M, r)
    U = construct_u(M, r, rows, columns)

    return C, U, R


####################################################################################################


def main():
    # predict_collaborative_filtering(md, ud, rd, pd)
    predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)

    # Save predictions, should be in the form 'list of tuples' or 'list of lists'
    with open(submission_file, 'w') as submission_writer:
        # Formats data
        predictions = [map(str, row) for row in predictions]
        predictions = [','.join(row) for row in predictions]
        predictions = 'Id,Rating\n'+'\n'.join(predictions)

        # Writes it down
        submission_writer.write(predictions)


def preprocessing():
    # util_matrix = construct_util_matrix(movies_description, users_description, ratings_description, predictions_description)
    C, U, R = cur(util_matrix, 6000)

    plt.imshow(util_matrix[:100, :100])
    plt.savefig('./M.png')

    plt.imshow(np.matmul(np.matmul(C, U), R)[:100, :100])
    plt.savefig('./CUR.png')


def matplotlib_test():
    M = np.matrix([
        [1, 1, 1, 0, 0],
        [3, 3, 3, 0, 0],
        [4, 4, 4, 0, 0],
        [5, 5, 5, 0, 0],
        [0, 0, 0, 4, 4],
        [0, 0, 0, 5, 5],
        [0, 0, 0, 2, 2]
    ])

    C, U, R = cur(M, 7)

    plt.imshow(M)
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(C)
    ax1 = fig.add_subplot(1,3,2)
    ax1.imshow(U)
    ax1 = fig.add_subplot(1,3,3)
    ax1.imshow(R)
    plt.show()

    plt.imshow(np.matmul(np.matmul(C, U), R))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    util_matrix = construct_util_matrix(movies_description, users_description, ratings_description, predictions_description, 100)

    M = global_baseline(util_matrix)
    plt.imshow(M)
    plt.colorbar()
    plt.show()
