import numpy as np
from sklearn.linear_model import orthogonal_mp_gram


def orthogonal_matching_pursuit(in_dictionary, in_y_matrix, in_k_sparse) -> np.ndarray:
    gram = in_dictionary.T.dot(in_dictionary)
    d_y = in_dictionary.T.dot(in_y_matrix)
    return orthogonal_mp_gram(gram, d_y, n_nonzero_coefs=in_k_sparse)
