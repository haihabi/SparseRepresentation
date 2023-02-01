import dictionary_learning
import netCDF4
import numpy as np
from dictionary_learning import metrics
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
experiment_type = ExperimentType.Approximation

data_cml = False
sum_threshold = 40

filter_data = experiment_type != ExperimentType.BatchLearning

if data_cml:
    data_folder = r"data\\"
    dataset.download_open_mrg(local_path=r"data\\")
    file_location = data_folder + "OpenMRG.zip"
    dataset.transform_open_mrg(file_location, data_folder)
    file_location = data_folder + f"radar\\radar.nc"
    radar_dataset = netCDF4.Dataset(file_location)
    data_list = []
    for i in tqdm(range(radar_dataset.variables['data'].shape[0])):
        x = np.asarray(radar_dataset.variables['data'][i, :, :])
        if np.all(x != 255):
            radar_rain_tensor = np.power(10, (x / 15)) * np.power(1 / 200, 1 / 1.5)
            if np.sum(radar_rain_tensor) > 40:
                patch_tensors = extract_patches_2d(radar_rain_tensor, (patch_size, patch_size))
                data_list.append(patch_tensors)

    data_raw = np.concatenate(data_list, axis=0)
else:
    data_raw = data.camera()
    data_raw = extract_patches_2d(data_raw, (patch_size, patch_size))  # Filter

np.random.shuffle(data_raw)
data_raw = data_raw.reshape([-1, patch_size ** 2]).astype("float")
# Split into training and validation
data_validation = data_raw[:validation_size, :]
data_training = data_raw[validation_size:, :]
if filter_data:
    data_training = data_training[:max_data_size, :]

mean_vector = np.mean(data_training, axis=0, keepdims=True)
data_training -= mean_vector
data_validation -= mean_vector
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
        k_svd = dictionary_learning.kSVD(n_iterations, sparsity, num_atoms=num_atoms, initialization_method=init_method)
        error = k_svd.train(data_training)
        plt.semilogy(error, label=init_method.name)
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.show()
#########################
# Ablation different MP
#########################
if experiment_type == ExperimentType.CodingMethod:

    for sparse_recovery_method in [
        dictionary_learning.SparseRecoveryMethod.MP,
        dictionary_learning.SparseRecoveryMethod.OMP,
    ]:
        k_svd = dictionary_learning.kSVD(n_iterations,
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
    plt.tight_layout()
    plt.show()

#########################
# Batch Training
#########################
if experiment_type == ExperimentType.BatchLearning:
    k_svd = dictionary_learning.kSVD(n_iterations,
                                     sparsity,
                                     num_atoms=num_atoms)
    error = k_svd.batch_training(data_training, batch_size=max_data_size, validation=data_validation)

    plt.semilogy(error, label="Batch Training")

    index = np.linspace(0, data_training.shape[0] - 1, data_training.shape[0]).astype("int")
    for i in range(5):
        np.random.shuffle(index)
        k_svd = dictionary_learning.kSVD(n_iterations,
                                         sparsity,
                                         num_atoms=num_atoms)
        error = k_svd.train(data_training[index[:max_data_size], :], validation=data_validation)
        plt.semilogy(error, label=f"Subset {i}")

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.show()

#########################
# Approximation kSVD
#########################
if experiment_type == ExperimentType.Approximation:
    k_svd = dictionary_learning.kSVD(n_iterations,
                                     sparsity,
                                     approx=True,
                                     num_atoms=num_atoms)
    error = k_svd.train(data_training, validation=data_validation)
    plt.semilogy(error, label=f"Approximated kSVD")

    k_svd = dictionary_learning.kSVD(n_iterations,
                                     sparsity,
                                     num_atoms=num_atoms)
    error = k_svd.train(data_training, validation=data_validation)

    plt.semilogy(error, label="kSVD")



    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.show()

# #########################
# # Ablation different N Atoms
# #########################
# mu_list = []
# error_list = []
# # 24, 28, 32, 64, 128
# num_atoms_array = [96, 128, 196, 256, 512, 1024]
# for num_atoms in num_atoms_array:
#     k_svd = dictionary_learning.kSVD(200, 8, num_atoms=num_atoms)
#     error = k_svd.train(data_matrix)
#
#     mu_bound = np.sqrt((num_atoms - d) / (d * (num_atoms - 1)))
#
#     mu = metrics.compute_mutual_coherence(k_svd.dictionary)
#     # print(mu, mu_bound)
#     # plt.subplot(1, 2, 1)
#     # dictionary_learning.plot_dictionary(k_svd.dictionary)
#     # plt.subplot(1, 2, 2)
#     # plt.semilogy(error)
#     # plt.show()
#
#     mu_list.append([mu, mu_bound])
#     error_list.append(error)
# mu_list = np.asarray(mu_list)
# error_list = np.asarray(error_list)
#
# # # plt.plot(mu_list)
# # # print(mu, mu_bound)
# # print(error_list.shape)
# for i, num_atoms in enumerate(num_atoms_array):
#     plt.semilogy(error_list[i, :], label=r"$N_{atoms}=$" + f"{num_atoms}")
# plt.legend()
# plt.grid()
# plt.xlabel("Iteration")
# plt.ylabel("Relative Error")
# plt.tight_layout()
# plt.show()
#
# #########################
# # reconsturction
# #########################
