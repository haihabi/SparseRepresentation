import numpy as np


def thresholding_pursuit(in_dictionary, in_y_matrix, in_k_sparse) -> np.ndarray:
    gammas = np.zeros((in_y_matrix.shape[1],in_dictionary.shape[1]))
    inners = np.abs(np.matmul(in_dictionary.T, in_y_matrix))
    idx = np.argsort(-inners.T)[:in_k_sparse, :in_k_sparse]
    gammas.T[idx] = inners[idx]
    return gammas.T
