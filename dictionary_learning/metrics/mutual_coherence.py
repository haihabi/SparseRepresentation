import numpy as np


def compute_mutual_coherence(in_d_matrix):
    n, K = in_d_matrix.shape
    mu = [np.abs(np.dot(in_d_matrix[:, i].T, in_d_matrix[:, j]) /
                 (np.linalg.norm(in_d_matrix[:, i]) * np.linalg.norm(in_d_matrix[:, j])))
          for i in range(K) for j in range(K) if j != i]
    return np.asarray(mu).max()

# def bound_
