import numpy as np
import dictionary_learning
from skimage.metrics import peak_signal_noise_ratio
from skimage import data
from skimage import restoration
import matplotlib.pyplot as plt
from applications.denosing.ksvd_denosing import ksvd_denoising


def plot_and_save_image(image, name):
    fig, ax = plt.subplots(figsize=[5, 5])
    ax.imshow(image, cmap='gray')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    axins = ax.inset_axes([0.65, 0.7, 0.4, 0.3])

    axins.imshow(image, cmap='gray')
    x1, x2, y1, y2 = 220, 270, 160, 90
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="red")
    plt.tight_layout()
    # plt.show()
    fig.savefig(name)


if __name__ == '__main__':

    np.random.seed(0)
    if False:
        for sigma in [15]:
            _res = []
            n_atoms_list = [64, 96, 128, 160, 192, 256, 512]
            for n_atoms in n_atoms_list:
                data_raw = data.camera().astype("float")
                image_noisy = data_raw + sigma * np.random.randn(*data_raw.shape)
                image_noisy = np.clip(image_noisy, a_min=0, a_max=255).astype("uint8").astype("float") / 255
                data_raw /= 255

                ksvd_config = dictionary_learning.KSVDConfig(15, 3, n_atoms,
                                                             in_sparse_recovery=dictionary_learning.SparseRecoveryMethod.MP,
                                                             initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS,
                                                             etol=1e-16,
                                                             approx=True)

                image_hat = ksvd_denoising(image_noisy, ksvd_config=ksvd_config, multiplier=0.0, patch_size=8,
                                           input_image2learn=data_raw)
                psnr = peak_signal_noise_ratio(data_raw, image_hat)
                _res.append(psnr)
            plt.plot(n_atoms_list, _res, label="$\sigma$=" + f"{sigma}")
        plt.legend()
        plt.grid()
        plt.xlabel(r"$d_x$")
        plt.ylabel("PSNR[dB]")
        plt.tight_layout()
        plt.savefig("n_atoms.svg")
    if False:
        for sigma in [15]:
            _res = []
            sparse_code_size = [1, 2, 3, 4, 5, 6, 7, 8]
            for scs in sparse_code_size:
                data_raw = data.camera().astype("float")
                image_noisy = data_raw + sigma * np.random.randn(*data_raw.shape)
                image_noisy = np.clip(image_noisy, a_min=0, a_max=255).astype("uint8").astype("float") / 255
                data_raw /= 255

                ksvd_config = dictionary_learning.KSVDConfig(15, scs, 180,
                                                             in_sparse_recovery=dictionary_learning.SparseRecoveryMethod.MP,
                                                             initialization_method=dictionary_learning.InitializationMethod.PLUSPLUS,
                                                             etol=1e-16,
                                                             approx=True)

                image_hat = ksvd_denoising(image_noisy, ksvd_config=ksvd_config, multiplier=0.0, patch_size=8,
                                           input_image2learn=data_raw)
                psnr = peak_signal_noise_ratio(data_raw, image_hat)
                _res.append(psnr)
            plt.plot(sparse_code_size, _res, label="$\sigma$=" + f"{sigma}")
        plt.legend()
        plt.grid()
        plt.xlabel(r"s")
        plt.ylabel("PSNR[dB]")
        plt.tight_layout()
        plt.savefig("scs.svg")
    if True:
        names = ["kSVD", "Wavelet", "BiLateral", "Non-Local-Means"]
        result = []
        sigma_array = np.linspace(10, 50, 20)
        # sigma_array = [50]
        for sigma in sigma_array:
            data_raw = data.camera().astype("float")
            image_noisy = data_raw + sigma * np.random.randn(*data_raw.shape)
            image_noisy = np.clip(image_noisy, a_min=0, a_max=255).astype("uint8").astype("float") / 255
            data_raw /= 255

            # plt.cla()
            # plt.clf()
            # plot_and_save_image(data_raw, f"data\images\\raw.png")

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
            image_list = [image_hat, image_hat_wave, image_hat_bilateral, image_hat_nlm]

            # plt.imshow(data_raw, cmap='gray')
            # plt.tight_layout()
            # plt.savefig(f"data\images\\raw.png")

            # for j, n in enumerate(names):
            #     plt.cla()
            #     plt.clf()
            #     plot_and_save_image(image_list[j], f"data\images\{n}_{sigma}.png")
            #     # plt.imshow(image_list[j], cmap='gray')
            #     # plt.savefig(f"data\images\{n}_{sigma}.png")

            psnr = peak_signal_noise_ratio(data_raw, image_hat)
            psnr_wavelet = peak_signal_noise_ratio(data_raw, image_hat_wave)
            psnr_bilateral = peak_signal_noise_ratio(data_raw, image_hat_bilateral)
            psnr_nlm = peak_signal_noise_ratio(data_raw, image_hat_nlm)
            result.append([psnr, psnr_wavelet, psnr_bilateral, psnr_nlm])

        result = np.asarray(result)
        for i in range(len(names)):
            plt.plot(sigma_array, result[:, i], "--" if names[i] == "kSVD" else "", label=names[i])
        plt.legend()
        plt.grid()
        plt.xlabel(r"$\sigma$")
        plt.ylabel("PSNR[dB]")
        plt.tight_layout()
        plt.savefig("image_vs_snr.svg")
        plt.show()
