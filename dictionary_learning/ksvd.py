import numpy as np
import tqdm

from enum import Enum
from sparselandtools.dictionaries.utils import overcomplete_haar_dictionary, overcomplete_idctii_dictionary
from sparselandtools.pursuits import MatchingPursuit
from dictionary_learning import sparse_recovery


class InitializationMethod(Enum):
    GAUSSIAN = 0
    DATA = 1
    PLUSPLUS = 2
    HAAR = 3
    DCT = 4


class SparseRecoveryMethod(Enum):
    MP = 0
    OMP = 1
    TMP = 2


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
                corr = in_y_matrix.T @ in_y_matrix
                norm_vec = np.linalg.norm(in_y_matrix, axis=0)
                cosin_distance = corr / (norm_vec.reshape([-1, 1]) @ norm_vec.reshape([1, -1]))
                corr_factor = np.sum(np.abs(cosin_distance), axis=0)
            else:
                corr = (in_y_matrix[:, index].T @ in_y_matrix[:, filter_list])
                norm_a = np.linalg.norm(in_y_matrix[:, index], axis=0)
                norm_b = np.linalg.norm(in_y_matrix[:, filter_list], axis=0)
                cosin_distance = corr / (norm_a.reshape([-1, 1]) @ norm_b.reshape([1, -1]))
                corr_factor = 1 / np.sum(np.abs(cosin_distance), axis=- 1)

            p = corr_factor / np.sum(corr_factor)
            select = np.random.choice(index, size=1, p=p)
            index = np.delete(index, np.where(select == index)[0])  # Remove from index list
            filter_list.append(select[0])
        y_matrix = in_y_matrix[:, index]
        d_matrix = in_y_matrix[:, filter_list] / np.linalg.norm(in_y_matrix[:, filter_list], axis=0)
    return d_matrix, y_matrix


class kSVD:
    def __init__(self, n_iterations: int,
                 k_sparse,
                 num_atoms,
                 in_sparse_recovery: SparseRecoveryMethod = SparseRecoveryMethod.OMP,
                 initialization_method: InitializationMethod = InitializationMethod.PLUSPLUS,
                 initial_dictionary=None,
                 etol=1e-10):
        self.n_iterations = n_iterations
        self.k_sparse = k_sparse
        self.initial_dictionary = initial_dictionary
        self.num_atoms = num_atoms
        self.initialization_method = initialization_method
        self.etol = etol
        self.sparse_recovery = in_sparse_recovery
        self.dictionary = None
        self.approx = False

    def initialized_dictionary(self, in_y_matrix: np.ndarray):
        y_matrix = in_y_matrix
        if self.initial_dictionary is not None:
            d_matrix = self.initial_dictionary / np.linalg.norm(self.initial_dictionary, axis=0)
        else:
            d_matrix, y_matrix = initialized_dictionary(in_y_matrix, self.num_atoms,
                                                        self.initialization_method)
        return d_matrix, y_matrix

    def sparse_coding(self, in_y_matrix, in_dictionary):
        if self.sparse_recovery == SparseRecoveryMethod.OMP:
            return sparse_recovery.orthogonal_matching_pursuit(in_dictionary,
                                                               in_y_matrix,
                                                               in_k_sparse=self.k_sparse)
        elif self.sparse_recovery == SparseRecoveryMethod.TMP:
            return sparse_recovery.thresholding_pursuit(in_dictionary,
                                                        in_y_matrix,
                                                        in_k_sparse=self.k_sparse)
        elif self.sparse_recovery == SparseRecoveryMethod.TMP:
            return sparse_recovery.thresholding_pursuit(in_dictionary,
                                                        in_y_matrix,
                                                        in_k_sparse=self.k_sparse)
        else:
            raise Exception("Unknown coding method")

    def update_dictionary(self, in_d_matrix, in_y_matrix, in_x_matrix):
        # codebook update stage
        for j in range(in_d_matrix.shape[1]):
            # index set of nonzero components
            index_set = np.nonzero(in_x_matrix[j, :])[0]
            if len(index_set) == 0:
                # for now, replace with some white noise
                if not self.approx:
                    in_d_matrix[:, j] = np.random.randn(*in_d_matrix[:, j].shape)
                    in_d_matrix[:, j] = in_d_matrix[:, j] / np.linalg.norm(in_d_matrix[:, j])
                continue

            if self.approx:
                # approximate K-SVD update
                error_matrix = Y[:, index_set] - in_d_matrix.dot(X[:, index_set])
                D[:, j] = error_matrix.dot(X[j, index_set])  # update D
                D[:, j] /= np.linalg.norm(D[:, j])
                X[j, index_set] = (error_matrix.T).dot(D[:, j])  # update X
            else:
                # error matrix E
                e_idx = np.delete(range(in_d_matrix.shape[1]), j, 0)
                error_matrix = in_y_matrix - np.dot(in_d_matrix[:, e_idx], in_x_matrix[e_idx, :])
                u_matrix, S, vt_matrix = np.linalg.svd(error_matrix[:, index_set])
                # update jth column of D
                in_d_matrix[:, j] = u_matrix[:, 0]
                # update sparse elements in jth row of X
                in_x_matrix[j, :] = np.array([
                    S[0] * vt_matrix[0, np.argwhere(index_set == n)[0][0]]
                    if n in index_set else 0
                    for n in range(in_x_matrix.shape[1])])
        return in_d_matrix

    def error_check(self, in_d_matrix, in_y_matrix, in_x_matrix):
        err = np.linalg.norm(in_y_matrix - in_d_matrix.dot(in_x_matrix), 'fro')
        norm_err = err / np.linalg.norm(in_y_matrix, 'fro')
        break_flag = norm_err < self.etol
        return err, break_flag

    def train(self, in_y_matrix):
        error_list = []
        if self.dictionary is None:
            dictionary, y_matrix = self.initialized_dictionary(in_y_matrix.T)
        else:
            dictionary = np.copy(self.dictionary)
            y_matrix = in_y_matrix
        for _ in tqdm.tqdm(range(self.n_iterations)):
            x_matrix = self.sparse_coding(y_matrix, dictionary)
            dictionary = self.update_dictionary(dictionary, y_matrix, x_matrix)
            error, break_flag = self.error_check(dictionary, y_matrix, x_matrix)
            error_list.append(error)
            if break_flag:
                break
        self.dictionary = dictionary
        return np.asarray(error_list)
