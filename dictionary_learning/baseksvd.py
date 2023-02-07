import numpy as np
import tqdm
from enum import Enum
from dictionary_learning import sparse_recovery
from dictionary_learning.config import KSVDConfig
from dictionary_learning.dictionary_initialization import InitializationMethod, initialized_dictionary
from dictionary_learning.sparse_recovery import SparseRecoveryMethod


class BaseKSVD:
    def __init__(self, n_iterations: int,
                 k_sparse,
                 num_atoms,
                 in_sparse_recovery: SparseRecoveryMethod = SparseRecoveryMethod.MP,
                 initialization_method: InitializationMethod = InitializationMethod.PLUSPLUS,
                 initial_dictionary=None,
                 etol=1e-10,
                 approx=False):
        self.dictionary_best = None
        self.n_iterations = n_iterations
        self.k_sparse = k_sparse
        self.initial_dictionary = initial_dictionary
        self.num_atoms = num_atoms
        self.initialization_method = initialization_method
        self.etol = etol
        self.sparse_recovery = in_sparse_recovery
        self.dictionary = None
        self.error_best = np.inf
        self.approx = approx

    def initialized_dictionary(self, in_y_matrix: np.ndarray):
        y_matrix = in_y_matrix
        if self.initial_dictionary is not None:
            d_matrix = self.initial_dictionary / np.linalg.norm(self.initial_dictionary, axis=0)
        else:
            d_matrix, y_matrix = initialized_dictionary(in_y_matrix, self.num_atoms,
                                                        self.initialization_method)
        return d_matrix, y_matrix

    def sparse_coding_step(self, in_y_matrix, in_dictionary, in_sparse_recovery=None):
        sparse_recovery_method = self.sparse_recovery if in_sparse_recovery is None else in_sparse_recovery

        if sparse_recovery_method == SparseRecoveryMethod.OMP:
            return sparse_recovery.orthogonal_matching_pursuit(in_dictionary,
                                                               in_y_matrix,
                                                               in_k_sparse=self.k_sparse)
        elif sparse_recovery_method == SparseRecoveryMethod.TMP:
            return sparse_recovery.thresholding_pursuit(in_dictionary,
                                                        in_y_matrix,
                                                        in_k_sparse=self.k_sparse)
        elif sparse_recovery_method == SparseRecoveryMethod.MP:
            return sparse_recovery.matching_pursuit(in_dictionary,
                                                    in_y_matrix,
                                                    in_k_sparse=self.k_sparse)
        else:
            raise Exception("Unknown coding method")

    def sparse_coding(self, in_y_matrix, in_sparse_recovery: SparseRecoveryMethod = None):
        return self.sparse_coding_step(in_y_matrix, self.dictionary_best, in_sparse_recovery=in_sparse_recovery)

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

                in_d_matrix[:, j] = 0
                error_matrix = in_y_matrix[:, index_set] - in_d_matrix.dot(in_x_matrix[:, index_set])
                d = error_matrix.dot(in_x_matrix[j, index_set])  # update D
                d /= np.linalg.norm(d)
                in_x_matrix[j, index_set] = error_matrix.T.dot(d)  # update X
                in_d_matrix[:, j] = d

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

    def get_dictionary(self, in_y_matrix):
        if self.dictionary is None:
            dictionary, y_matrix = self.initialized_dictionary(in_y_matrix.T)
        else:
            dictionary = np.copy(self.dictionary)
            y_matrix = in_y_matrix.T
        return dictionary, y_matrix

    def train_step(self, dictionary, y_matrix):
        x_matrix = self.sparse_coding_step(y_matrix, dictionary)
        dictionary = self.update_dictionary(dictionary, y_matrix, x_matrix)
        return dictionary, x_matrix

    def compute_error(self, dictionary, validation, y_matrix, x_matrix):
        if validation is not None:
            validation = validation.T
            x_matrix_val = self.sparse_coding_step(validation, dictionary)
            error, break_flag = self.error_check(dictionary, validation, x_matrix_val)
        else:
            error, break_flag = self.error_check(dictionary, y_matrix, x_matrix)
        return error, break_flag

    def train(self, in_y_matrix,
              n_iteration=None,
              validation=None,
              batch_training=False):
        n_iteration = self.n_iterations if n_iteration is None else n_iteration
        error_list = []
        dictionary, y_matrix = self.get_dictionary(in_y_matrix)
        if not batch_training:
            pbar = tqdm.tqdm(total=n_iteration)

        for _ in range(n_iteration):
            dictionary, x_matrix = self.train_step(dictionary, y_matrix)
            error, break_flag = self.compute_error(dictionary, validation, y_matrix, x_matrix)
            if error < self.error_best:
                self.error_best = error
                self.dictionary_best = dictionary
            if not batch_training:
                pbar.set_description("kSVD Training loop progress")
                pbar.set_postfix({"Relative Error": error})
                pbar.update(1)
                error_list.append(error)
                if break_flag:
                    break
        self.dictionary = dictionary
        if not batch_training:
            pbar.close()
            return np.asarray(error_list)
        else:
            return error, break_flag

    def batch_training(self, in_y_matrix, batch_size, n_iteration=None, validation=None):
        self.dictionary, _y_matrix = self.get_dictionary(in_y_matrix)
        n_iteration = self.n_iterations if n_iteration is None else n_iteration
        pbar = tqdm.tqdm(total=n_iteration)
        index = np.linspace(0, _y_matrix.shape[0] - 1, _y_matrix.shape[0]).astype("int")
        error_list = []
        for _ in range(n_iteration):
            np.random.shuffle(index)
            y_matrix = _y_matrix[:, index[:batch_size]]  # Take a random batch of samples

            error, break_flag = self.train(y_matrix.T, n_iteration=1, batch_training=True, validation=validation)
            error_list.append(error)
            if break_flag:
                break
            pbar.set_description("Batch kSVD Training loop progress")
            pbar.set_postfix({"Relative Error": error, "Relative Error Best": self.error_best})
            pbar.update(1)
        pbar.close()
        return np.asarray(error_list)


class KSVD(BaseKSVD):
    def __init__(self, ksvd_config: KSVDConfig):
        super().__init__(ksvd_config.n_iterations,
                         ksvd_config.k_sparse,
                         ksvd_config.num_atoms,
                         ksvd_config.in_sparse_recovery,
                         ksvd_config.initialization_method,
                         ksvd_config.initial_dictionary,
                         ksvd_config.etol,
                         ksvd_config.approx)
