import numpy as np


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
