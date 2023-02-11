import dictionary_learning
import netCDF4
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from skimage import data
import dataset
from enum import Enum


class ExperimentType(Enum):
    Initialization = 0
    CodingMethod = 1
    BatchLearning = 2
    Approximation = 3


patch_size = 8

max_data_size = 2048
validation_size = 2048
num_atoms = 128
n_iterations = 200
sparsity = 8
experiment_type = ExperimentType.Initialization

data_cml = True
sum_threshold = 40

filter_data = experiment_type != ExperimentType.BatchLearning

data_training, data_validation, mean_vector = dataset.get_dataset(data_cml, patch_size, filter_data, validation_size,
                                                                  max_data_size)
#########################
# Ablation different Initilization
#########################
if experiment_type == ExperimentType.Initialization:

    for init_method in [
        dictionary_learning.InitializationMethod.PLUSPLUS,
        dictionary_learning.InitializationMethod.DCT,
        dictionary_learning.InitializationMethod.HAAR,
        dictionary_learning.InitializationMethod.GAUSSIAN,
        dictionary_learning.InitializationMethod.DATA,
    ]:
        k_svd = dictionary_learning.BaseKSVD(n_iterations, sparsity, num_atoms=num_atoms,
                                             initialization_method=init_method)
        error = k_svd.train(data_training)
        plt.semilogy(error, label=init_method.name)
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")

#########################
# Ablation different MP
#########################
if experiment_type == ExperimentType.CodingMethod:

    for sparse_recovery_method in [
        dictionary_learning.SparseRecoveryMethod.MP,
        dictionary_learning.SparseRecoveryMethod.OMP,
    ]:
        k_svd = dictionary_learning.BaseKSVD(n_iterations,
                                             sparsity,
                                             num_atoms=num_atoms,
                                             in_sparse_recovery=sparse_recovery_method,
                                             initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS)
        error = k_svd.train(data_training)
        plt.semilogy(error, label=sparse_recovery_method.name)
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")

#########################
# Batch Training
#########################
if experiment_type == ExperimentType.BatchLearning:
    k_svd = dictionary_learning.BaseKSVD(n_iterations,
                                         sparsity,
                                         num_atoms=num_atoms)
    error = k_svd.batch_training(data_training, batch_size=max_data_size, validation=data_validation)

    plt.semilogy(error, label="Batch Training")

    index = np.linspace(0, data_training.shape[0] - 1, data_training.shape[0]).astype("int")
    for i in range(5):
        np.random.shuffle(index)
        k_svd = dictionary_learning.BaseKSVD(n_iterations,
                                             sparsity,
                                             num_atoms=num_atoms)
        error = k_svd.train(data_training[index[:max_data_size], :], validation=data_validation)
        plt.semilogy(error, label=f"Subset {i}")

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")

#########################
# Approximation kSVD
#########################
if experiment_type == ExperimentType.Approximation:
    k_svd = dictionary_learning.BaseKSVD(n_iterations,
                                         sparsity,
                                         approx=True,
                                         num_atoms=num_atoms)
    error = k_svd.train(data_training, validation=data_validation)
    plt.semilogy(error, label=f"Approximated kSVD")

    k_svd = dictionary_learning.BaseKSVD(n_iterations,
                                         sparsity,
                                         num_atoms=num_atoms)
    error = k_svd.train(data_training, validation=data_validation)

    plt.semilogy(error, label="kSVD")

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
plt.tight_layout()
data_name = "cml" if data_cml else "image"
plt.savefig(f"{experiment_type.name}_{data_name}.svg")
plt.show()
