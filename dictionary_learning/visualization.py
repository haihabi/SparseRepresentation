import numpy as np
from matplotlib import pyplot as plt


def plot_dictionary(in_d_matrix):
    # d_matrix =
    n, K = in_d_matrix.shape
    M = np.copy(in_d_matrix)
    # # stretch atoms
    for k in range(K):
        M[:, k] = M[:, k] - (M[:, k].min())
        if M[:, k].max():
            M[:, k] = M[:, k] / M[:, k].max()

    # patch size
    n_r = int(np.sqrt(n))

    # patches per row / column
    K_r = int(np.sqrt(K))

    # we need n_r*K_r+K_r+1 pixels in each direction
    dim = n_r * K_r + K_r + 1
    V = np.ones((dim, dim)) * np.min(M)

    # compute the patches
    patches = [np.reshape(M[:, i], (n_r, n_r)) for i in range(K)]

    # place patches
    for i in range(K_r):
        for j in range(K_r):
            V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                i * K_r + j]

    plt.imshow(V, cmap='gray')
    plt.axis('off')
    # plt.colorbar()
    # plt.show()
