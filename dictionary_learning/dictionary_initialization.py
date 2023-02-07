import numpy as np
from enum import Enum
from sparselandtools.dictionaries.utils import overcomplete_haar_dictionary, overcomplete_idctii_dictionary
from dictionary_learning.metrics.mutual_coherence import compute_mutual_coherence


class InitializationMethod(Enum):
    GAUSSIAN = 0
    DATA = 1
    PLUSPLUS = 2
    HAAR = 3
    DCT = 4


def compute_cosine_distance(in_matrix_a, in_matrix_b):
    corr = in_matrix_a.T @ in_matrix_b
    norm_a = np.linalg.norm(in_matrix_a, axis=0)
    norm_b = np.linalg.norm(in_matrix_b, axis=0)
    return corr / (norm_a.reshape([-1, 1]) @ norm_b.reshape([1, -1]))


def initialized_dictionary(in_y_matrix: np.ndarray, num_atoms: int, initialization_method: InitializationMethod):
    y_matrix = in_y_matrix
    # randomly select initial dictionary from data
    if initialization_method == InitializationMethod.DATA:
        idx_set = range(in_y_matrix.shape[1])
        idxs = np.random.choice(idx_set, num_atoms, replace=False)
        y_matrix = in_y_matrix[:, np.delete(idx_set, idxs)]
        d_matrix = in_y_matrix[:, idxs] / np.linalg.norm(in_y_matrix[:, idxs], axis=0)
    elif initialization_method == InitializationMethod.GAUSSIAN:
        d_matrix = np.random.randn(in_y_matrix.shape[0], num_atoms)
        d_matrix = d_matrix / np.linalg.norm(d_matrix, axis=0, keepdims=True)
    elif initialization_method == InitializationMethod.HAAR:
        d_matrix = overcomplete_haar_dictionary(np.sqrt(in_y_matrix.shape[0]).astype("int"), num_atoms).T[
                   :, :num_atoms]
    elif initialization_method == InitializationMethod.DCT:
        d_matrix = overcomplete_idctii_dictionary(np.sqrt(in_y_matrix.shape[0]).astype("int"),
                                                  num_atoms)[
                   :, :num_atoms]
    elif initialization_method == InitializationMethod.PLUSPLUS:
        index = np.linspace(0, in_y_matrix.shape[1] - 1, in_y_matrix.shape[1]).astype("int")
        filter_list = []
        for i in range(num_atoms):
            if i == 0:
                subset_index = np.random.choice(index, 20)
                y_sub_matrix = in_y_matrix[:, subset_index]
                cosine_distance = compute_cosine_distance(y_sub_matrix, in_y_matrix)
                corr_factor = np.mean(np.abs(cosine_distance), axis=-1)
            else:
                subset_index = index
                cosine_distance = compute_cosine_distance(in_y_matrix[:, index], in_y_matrix[:, filter_list])
                corr_factor = 1 - np.max(np.abs(cosine_distance), axis=-1)

            p = corr_factor / np.sum(corr_factor)
            select = np.random.choice(subset_index, size=1, p=p)
            index = np.delete(index, np.where(select == index)[0])  # Remove from index list
            filter_list.append(select[0])
        y_matrix = in_y_matrix[:, index]
        d_matrix = in_y_matrix[:, filter_list] / np.linalg.norm(in_y_matrix[:, filter_list], axis=0)
    return d_matrix, y_matrix
