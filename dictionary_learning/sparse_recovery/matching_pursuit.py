import numpy as np


def matching_pursuit(in_dictionary, in_y_matrix, in_k_sparse, max_iter=100) -> np.ndarray:
    # analyze shape of Y
    if len(in_y_matrix.shape) == 1:
        data = np.array([in_y_matrix])
    elif len(in_y_matrix.shape) == 2:
        data = in_y_matrix
    else:
        raise ValueError("Input must be a vector or a matrix.")
    # analyze dimensions
    if not in_dictionary.shape[0] == in_y_matrix.shape[0]:
        raise ValueError("Dimension mismatch: %s != %s" % (in_dictionary.shape[0], in_y_matrix.shape[0]))

    alphas = []
    for y in data.T:
        # temporary values
        coeffs = np.zeros(in_dictionary.shape[1])
        residual = y

        # iterate
        i = 0
        if max_iter:
            m = max_iter
        else:
            m = np.inf

        finished = False

        while not finished:
            if i >= m:
                break
            inner = np.dot(in_dictionary.T, residual)
            gamma = int(np.argmax(np.abs(inner)))
            alpha = inner[gamma]
            residual = residual - alpha * in_dictionary[:, gamma]
            if np.isclose(alpha, 0):
                break
            coeffs[gamma] += alpha
            i += 1
            finished = np.count_nonzero(coeffs) >= in_k_sparse

        alphas.append(coeffs)
    return np.transpose(alphas)
