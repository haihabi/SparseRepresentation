import dictionary_learning
# import dataset
import netCDF4
import numpy as np
from dictionary_learning import metrics
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from skimage import data
import dataset

patch_size = 8

sum_threshold = 40
d = patch_size ** 2
max_patches = 2048
data_cml = True
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
                patch_tensors = extract_patches_2d(radar_rain_tensor, (patch_size, patch_size), max_patches=max_patches)
                data_list.append(patch_tensors)

    data_matrix = np.concatenate(data_list, axis=0)
    np.random.shuffle(data_matrix)
    data_matrix = data_matrix[:max_patches, :]
    data_matrix = data_matrix.reshape(max_patches, -1)

else:
    data_matrix = data.camera()
    data_matrix = extract_patches_2d(data_matrix, (patch_size, patch_size), max_patches=max_patches)
    data_matrix = data_matrix.reshape([-1, patch_size ** 2]).astype("float")
    # print("a")
data_matrix -= np.mean(data_matrix, axis=0, keepdims=True)
#########################
# Ablation different Initilization
#########################
if False:
    num_atoms = 128

    for init_method in [
        dictionary_learning.InitializationMethod.PLUSPLUS,
        dictionary_learning.InitializationMethod.DCT,
        dictionary_learning.InitializationMethod.HAAR,
        dictionary_learning.InitializationMethod.GAUSSIAN,
        dictionary_learning.InitializationMethod.DATA,
    ]:
        k_svd = dictionary_learning.kSVD(200, 8, num_atoms=num_atoms, initialization_method=init_method)
        error = k_svd.train(data_matrix)
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
if True:
    num_atoms = 128
    for init_method in [
        dictionary_learning.SparseRecoveryMethod.MP,
        dictionary_learning.SparseRecoveryMethod.OMP,
        dictionary_learning.SparseRecoveryMethod.TMP,

    ]:
        k_svd = dictionary_learning.kSVD(200,
                                         8,
                                         num_atoms=num_atoms,
                                         initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS)
        error = k_svd.train(data_matrix)
        plt.semilogy(error, label=init_method.name)
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.show()


#########################
# Ablation different N Atoms
#########################
mu_list = []
error_list = []
# 24, 28, 32, 64, 128
num_atoms_array = [96, 128, 196, 256, 512, 1024]
for num_atoms in num_atoms_array:
    k_svd = dictionary_learning.kSVD(200, 8, num_atoms=num_atoms)
    error = k_svd.train(data_matrix)

    mu_bound = np.sqrt((num_atoms - d) / (d * (num_atoms - 1)))

    mu = metrics.compute_mutual_coherence(k_svd.dictionary)
    # print(mu, mu_bound)
    # plt.subplot(1, 2, 1)
    # dictionary_learning.plot_dictionary(k_svd.dictionary)
    # plt.subplot(1, 2, 2)
    # plt.semilogy(error)
    # plt.show()

    mu_list.append([mu, mu_bound])
    error_list.append(error)
mu_list = np.asarray(mu_list)
error_list = np.asarray(error_list)

# # plt.plot(mu_list)
# # print(mu, mu_bound)
# print(error_list.shape)
for i, num_atoms in enumerate(num_atoms_array):
    plt.semilogy(error_list[i, :], label=r"$N_{atoms}=$" + f"{num_atoms}")
plt.legend()
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Relative Error")
plt.tight_layout()
plt.show()

# plt.plot(num_atoms_array, mu_list[:, 0])
# plt.plot(num_atoms_array, mu_list[:, 1])
# plt.grid()
# plt.show()
#
# plt.plot(error)
# plt.show()

#########################
# reconsturction
#########################
