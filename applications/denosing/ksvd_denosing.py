import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d
import dictionary_learning
from dictionary_learning.config import KSVDConfig


def prepare_k_svd(input_image, patch_size, remove_mean=True):
    patches = extract_patches_2d(input_image, (patch_size, patch_size))
    patches = patches.reshape([-1, patch_size ** 2])
    mean_vector = np.mean(patches, axis=0, keepdims=True)
    if remove_mean:
        patches -= mean_vector
    return patches, mean_vector


def ksvd_denoising(input_image,
                   multiplier=5,
                   ksvd_config: KSVDConfig = KSVDConfig.get_default_config(),
                   batch_size=4096,
                   patch_size=8,
                   input_image2learn=None):
    # prepare K-SVD
    patches, mean_vector = prepare_k_svd(input_image, patch_size, remove_mean=input_image2learn is None)

    k_svd_learner = dictionary_learning.KSVD(ksvd_config)
    if input_image2learn is None:
        k_svd_learner.batch_training(patches, batch_size)
    else:
        patches_clean, mean_vector = prepare_k_svd(input_image2learn, patch_size)
        patches -= mean_vector
        k_svd_learner.batch_training(patches_clean, batch_size)

    # reconstruct image
    # this was translated from the Matlab code in Michael Elads book
    # cf. Elad, M. (2010). Sparse and redundant representations:
    # from theory to applications in signal and image processing. New York: Springer.
    out = np.zeros(input_image.shape)
    weight = np.zeros(input_image.shape)
    x_matrix = k_svd_learner.sparse_coding(patches.T)
    patches_clean = np.reshape(np.matmul(k_svd_learner.dictionary_best, x_matrix) + mean_vector.T,
                               (patch_size, patch_size, -1))

    i = j = 0
    for k in range((input_image.shape[0] - patch_size + 1) ** 2):
        patch = patches_clean[:, :, k]
        out[j:j + patch_size, i:i + patch_size] += patch
        weight[j:j + patch_size, i:i + patch_size] += 1
        if i < input_image.shape[0] - patch_size:
            i += 1
        else:
            i = 0
            j += 1
    return np.clip(np.divide(out + multiplier * input_image, weight + multiplier), a_min=0, a_max=1)
