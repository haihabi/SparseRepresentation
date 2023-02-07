import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d
import dictionary_learning
from dictionary_learning.config import KSVDConfig
from skimage.metrics import peak_signal_noise_ratio

data_base = np.asarray(
    [[30.772452920813205, 28.296367510182392, 30.205351093738194, 25.576837617948343, 30.076504428517353],
     [30.06192588030512, 28.303004262223062, 27.79506689049653, 25.43837227894942, 30.050505754864325],
     [29.35111109123465, 28.261640616346618, 26.85255805261372, 25.134915341840983, 29.982799249178036],
     [27.936516653220202, 28.192551365253635, 25.579377706168014, 24.874806003047375, 29.72048961875204],
     [27.282043340699232, 28.066061351646738, 24.462015807649063, 24.40546261357372, 29.29041514120768],
     [26.43614186712537, 27.80530132181726, 24.33531073950111, 23.844797335903884, 28.384335845179717],
     [25.61522593756928, 27.358953071937112, 23.617412679981395, 23.234499028097204, 26.679007430420523],
     [24.563662901927085, 26.46501023430599, 23.094222449107697, 22.568106885810757, 24.004917014026013],
     [24.216322614857233, 26.151331939961096, 22.62432110493811, 21.843266410183745, 20.422511928035625],
     [23.711408814165893, 25.552171989895662, 21.91494776275274, 21.170611871822214, 17.276047357646632]])


def prepare_k_svd(input_image, patch_size, remove_mean=True):
    patches = extract_patches_2d(input_image, (patch_size, patch_size))
    patches = patches.reshape([-1, patch_size ** 2])
    mean_vector = np.mean(patches, axis=0, keepdims=True)
    if remove_mean:
        patches -= mean_vector
    return patches, mean_vector


#
#
# def shrinkage_function(in_x, in_lambda, in_s):
#     x_abs = np.abs(in_x)
#     _x_diff = x_abs - in_lambda - in_s
#     out = (_x_diff + np.sqrt(_x_diff ** 2 + 4 * in_s * x_abs)) / 2
#     return np.sign(in_x) * out
#
#     # t = np.abs(in_x) - in_spares * np.log(1 + np.abs(in_x) / in_spares)
#     # ind = np.abs(in_x) < t
# return (np.abs(in_x) >= t) * (np.abs(in_x) - t) * np.sign(in_x)


# def kvsd_denoising_v2(input_image,
#                       multiplier=5,
#                       ksvd_config: KSVDConfig = KSVDConfig.get_default_config(),
#                       batch_size=4096,
#                       patch_size=8,
#                       clean_image=None):
#     patches, mean_vector = prepare_k_svd(input_image, patch_size, clean_image is None)
#     k_svd_learner = dictionary_learning.KSVD(ksvd_config)
#     if clean_image is not None:
#         patches_clean, mean_vector = prepare_k_svd(clean_image, patch_size)
#         patches -= mean_vector
#         k_svd_learner.batch_training(patches, patches_clean)
#     else:
#         k_svd_learner.batch_training(patches, batch_size)
#
#     d_matrix = k_svd_learner.dictionary
#     # x_matrix = soft_threshold(np.linalg.inv(d_matrix.T @ d_matrix) @ d_matrix.T @ patches.T, 0.01)
#
#     x_matrix = np.zeros([d_matrix.shape[1], patches.shape[0]])
#     for i in range(10):
#         error = patches.T - d_matrix @ x_matrix
#         x_matrix = shrinkage_function(x_matrix + d_matrix.T @ error, 0.075, 0.01)
#
#     print("a")


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
        # patches, mean_vector = prepare_k_svd(input_image, patch_size)
        # k_svd_learner.batch_training(patches, batch_size)
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


if __name__ == '__main__':
    from skimage import data
    from skimage import restoration
    import matplotlib.pyplot as plt

    np.random.seed(0)

    result = []
    sigma_array = np.linspace(10, 50, 3)
    for sigma in sigma_array:
        data_raw = data.camera().astype("float")
        image_noisy = data_raw + sigma * np.random.randn(*data_raw.shape)
        image_noisy = np.clip(image_noisy, a_min=0, a_max=255).astype("uint8").astype("float") / 255
        data_raw /= 255

        ksvd_config = KSVDConfig(4000, 3, 180,
                                 in_sparse_recovery=dictionary_learning.SparseRecoveryMethod.MP,
                                 initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS,
                                 etol=1e-16,
                                 approx=True)

        # image_hat_v2 = kvsd_denoising_v2(image_noisy, ksvd_config=ksvd_config, multiplier=0.0, patch_size=6,
        #                                  clean_image=data_raw)
        image_hat = ksvd_denoising(image_noisy, ksvd_config=ksvd_config, multiplier=0.0, patch_size=8,
                                   input_image2learn=data_raw)
        image_hat_wave = restoration.denoise_wavelet(image_noisy, mode="hard", wavelet="haar")
        image_hat_bilateral = restoration.denoise_bilateral(image_noisy)
        image_hat_nlm = restoration.denoise_nl_means(image_noisy, patch_size=4, patch_distance=4)
        image_hat_tv = restoration.denoise_tv_chambolle(image_noisy, n_iter_max=20)

        psnr = peak_signal_noise_ratio(data_raw, image_hat)
        psnr_wavelet = peak_signal_noise_ratio(data_raw, image_hat_wave)
        psnr_tv = peak_signal_noise_ratio(data_raw, image_hat_tv)
        psnr_bilateral = peak_signal_noise_ratio(data_raw, image_hat_bilateral)
        psnr_nlm = peak_signal_noise_ratio(data_raw, image_hat_nlm)
        result.append([psnr, psnr_tv, psnr_wavelet, psnr_bilateral, psnr_nlm])
        #
        # print(result[-1])
    print(result)
    names = ["kSVD", "TV", "Wavelet", "BiLateral", "Non-Local-Means"]
    result = np.asarray(result)
    for i in range(5):
        plt.plot(sigma_array, result[:, i], label=names[i])
        print(result[:, i])
    # plt.plot(sigma_array, result, "--", label="kSVD")
    plt.legend()
    plt.grid()
    plt.show()
