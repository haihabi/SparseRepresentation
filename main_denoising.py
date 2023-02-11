import numpy as np
import dictionary_learning
from skimage.metrics import peak_signal_noise_ratio
from skimage import data
from skimage import restoration
import matplotlib.pyplot as plt
from applications.denosing.ksvd_denosing import ksvd_denoising

if __name__ == '__main__':

    np.random.seed(0)

    result = []
    sigma_array = np.linspace(10, 50, 20)
    for sigma in sigma_array:
        data_raw = data.camera().astype("float")
        image_noisy = data_raw + sigma * np.random.randn(*data_raw.shape)
        image_noisy = np.clip(image_noisy, a_min=0, a_max=255).astype("uint8").astype("float") / 255
        data_raw /= 255

        ksvd_config = dictionary_learning.KSVDConfig(15, 3, 180,
                                                     in_sparse_recovery=dictionary_learning.SparseRecoveryMethod.MP,
                                                     initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS,
                                                     etol=1e-16,
                                                     approx=True)

        image_hat = ksvd_denoising(image_noisy, ksvd_config=ksvd_config, multiplier=0.0, patch_size=8,
                                   input_image2learn=data_raw)
        image_hat_wave = restoration.denoise_wavelet(image_noisy, mode="hard", wavelet="haar")
        image_hat_bilateral = restoration.denoise_bilateral(image_noisy)
        image_hat_nlm = restoration.denoise_nl_means(image_noisy, patch_size=3, patch_distance=4)

        psnr = peak_signal_noise_ratio(data_raw, image_hat)
        psnr_wavelet = peak_signal_noise_ratio(data_raw, image_hat_wave)
        psnr_bilateral = peak_signal_noise_ratio(data_raw, image_hat_bilateral)
        psnr_nlm = peak_signal_noise_ratio(data_raw, image_hat_nlm)
        result.append([psnr, psnr_wavelet, psnr_bilateral, psnr_nlm])

    names = ["kSVD", "Wavelet", "BiLateral", "Non-Local-Means"]
    result = np.asarray(result)
    for i in range(len(names)):
        plt.plot(sigma_array, result[:, i], "--" if names[i] == "kSVD" else "", label=names[i])
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\sigma$")
    plt.ylabel("PSNR[dB]")
    plt.tight_layout()
    plt.show()
