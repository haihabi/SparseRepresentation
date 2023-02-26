import matplotlib.pyplot as plt

import dataset
import numpy as np
import dictionary_learning

data_cml = True
patch_size = -1
filter_data = False
max_data_size = np.inf
validation_size = 512
from matplotlib import pyplot as plt

np.random.seed(0)


class KSVDRainReconstricution:
    def __init__(self, in_sampler, in_ksvd_learner, in_mean_vector):
        self.d_hat = in_sampler.h_matrix @ in_ksvd_learner.dictionary_best
        self.ksvd_learner = in_ksvd_learner
        self.s = in_ksvd_learner.k_sparse
        self.mean_vector = in_mean_vector

    def reconstruct(self, in_y):
        x = dictionary_learning.sparse_recovery.matching_pursuit(self.d_hat, np.expand_dims(in_y, axis=1), self.s)
        r_hat = self.ksvd_learner.dictionary_best @ x + self.mean_vector.T
        u = int(np.sqrt(np.prod(r_hat.shape)))
        r_hat = r_hat.reshape([u, u])
        return np.maximum(r_hat, 0)


def validation(in_estimator: KSVDRainReconstricution, add2name="", plot_maps=False):
    res = 0
    for i in range(data_validation.shape[0]):
        rain_field = np.reshape(data_validation[i, :], [16, 16])
        y = sampler.sample(rain_field)
        r_hat = in_estimator.reconstruct(y)

        if i < 1024 and plot_maps:
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=[5, 5])
            t = ax.imshow(rain_field)

            # plt.imshow(rain_field)
            # ax.plot(sampler.x_mid, sampler.y_mid, "o", color="red")
            fig.colorbar(t)
            # plt.title("Ground Truth")
            plt.xlabel("x[km]")
            plt.ylabel("y[km]")
            plt.tight_layout()

            plt.savefig(f"data\images_rain\\rain_example_{i}_gt_{add2name}.png")
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=[5, 5])
            t = ax.imshow(r_hat)
            # plt.imshow(r_hat)
            fig.colorbar(t)

            plt.xlabel("x[km]")
            plt.ylabel("y[km]")
            plt.tight_layout()
            plt.savefig(f"data\images_rain\\rain_example_{i}_{add2name}.png")
        res += np.mean(np.power(r_hat - rain_field, 2.0))
    return res / data_validation.shape[0]


class Sampler:
    def __init__(self, point_sensors, line_sensors, in_shape, m=100):
        point_enable = line_enable = False
        if line_sensors is not None and line_sensors.shape[0] > 0:
            line_enable = True
            self.line_sensors = line_sensors
            h_matrix_line = np.zeros([self.line_sensors.shape[0], np.prod(in_shape)])
            h_matrix_count = np.zeros([self.line_sensors.shape[0], np.prod(in_shape)])

            self.x_one_line = self.line_sensors[:, 0].reshape([-1, 1]) * 15
            self.y_one_line = self.line_sensors[:, 1].reshape([-1, 1]) * 15
            self.x_two_line = self.line_sensors[:, 2].reshape([-1, 1]) * 15
            self.y_two_line = self.line_sensors[:, 3].reshape([-1, 1]) * 15
            s = np.linspace(0, 1, m).reshape([1, -1])

            x_mid = self.x_one_line + (self.x_two_line - self.x_one_line) * s
            y_mid = self.y_one_line + (self.y_two_line - self.y_one_line) * s

            x_c = np.ceil(x_mid).astype("int")
            x_f = np.floor(x_mid).astype("int")
            y_c = np.ceil(y_mid).astype("int")
            y_f = np.floor(y_mid).astype("int")

            s_x = x_mid / x_c
            s_y = y_mid / y_c

            cc_index_flat = np.ravel_multi_index(np.stack([x_c.flatten(), y_c.flatten()]), in_shape).reshape([-1, m])
            cf_index_flat = np.ravel_multi_index(np.stack([x_c.flatten(), y_f.flatten()]), in_shape).reshape([-1, m])
            fc_index_flat = np.ravel_multi_index(np.stack([x_f.flatten(), y_c.flatten()]), in_shape).reshape([-1, m])
            ff_index_flat = np.ravel_multi_index(np.stack([x_f.flatten(), y_f.flatten()]), in_shape).reshape([-1, m])
            for i in range(self.line_sensors.shape[0]):
                for j in range(m):
                    h_matrix_line[i, cc_index_flat[i, j]] += s_x[i, j] * s_y[i, j] / m
                    h_matrix_line[i, cf_index_flat[i, j]] += s_x[i, j] * (1 - s_y[i, j]) / m
                    h_matrix_line[i, fc_index_flat[i, j]] += (1 - s_x[i, j]) * s_y[i, j] / m
                    h_matrix_line[i, ff_index_flat[i, j]] += (1 - s_x[i, j]) * (1 - s_y[i, j]) / m

                    h_matrix_count[i, cc_index_flat[i, j]] += 1
                    h_matrix_count[i, cf_index_flat[i, j]] += 1
                    h_matrix_count[i, fc_index_flat[i, j]] += 1
                    h_matrix_count[i, ff_index_flat[i, j]] += 1
            h_matrix_line = (h_matrix_line / (h_matrix_count + 1e-6)) * ((h_matrix_count > 0).astype("float"))

        if point_sensors is not None and point_sensors.shape[0] > 0:
            point_enable = True
            self.point_sensors = point_sensors
            h_matrix_point = np.zeros([self.point_sensors.shape[0], np.prod(in_shape)])
            self.x_mid = self.point_sensors[:, 0] * 15
            self.y_mid = self.point_sensors[:, 1] * 15
            self.x_c = np.ceil(self.x_mid).astype("int")
            self.x_f = np.floor(self.x_mid).astype("int")
            self.y_c = np.ceil(self.y_mid).astype("int")
            self.y_f = np.floor(self.y_mid).astype("int")

            self.s_x = self.x_mid / self.x_c
            self.s_y = self.y_mid / self.y_c

            cc_index_flat = np.ravel_multi_index(np.stack([self.x_c, self.y_c]), in_shape)
            cf_index_flat = np.ravel_multi_index(np.stack([self.x_c, self.y_f]), in_shape)
            fc_index_flat = np.ravel_multi_index(np.stack([self.x_f, self.y_c]), in_shape)
            ff_index_flat = np.ravel_multi_index(np.stack([self.x_f, self.y_f]), in_shape)

            for i in range(self.point_sensors.shape[0]):
                h_matrix_point[i, cc_index_flat[i]] = self.s_x[i] * self.s_y[i]
                h_matrix_point[i, cf_index_flat[i]] = self.s_x[i] * (1 - self.s_y[i])
                h_matrix_point[i, fc_index_flat[i]] = (1 - self.s_x[i]) * self.s_y[i]
                h_matrix_point[i, ff_index_flat[i]] = (1 - self.s_x[i]) * (1 - self.s_y[i])
        if line_enable and point_enable:
            h_matrix = np.stack([h_matrix_line, h_matrix_point], axis=0)
        elif line_enable:
            h_matrix = h_matrix_line
        elif point_enable:
            h_matrix = h_matrix_point
        else:
            raise Exception("AA")

        self.h_matrix = h_matrix

    def sample(self, in_field):
        return self.h_matrix @ in_field.flatten()


def generate_sensors(n_p, n_l, min_point=-5, max_point=5):
    s = np.random.rand(n_p, 2)
    s_link = np.random.rand(n_l, 4)
    return Sampler(s, s_link, (16, 16))


if __name__ == '__main__':
    data_training, data_validation, mean_vector = dataset.get_dataset(data_cml, patch_size, filter_data,
                                                                      validation_size,
                                                                      max_data_size)
    ksvd_config = dictionary_learning.KSVDConfig(200, 30, 512)
    ksvd_learner = dictionary_learning.KSVD(ksvd_config)
    ksvd_learner.batch_training(data_training, batch_size=256)

    if True:
        m = 100
        n_sensors_list = [5, 15, 30]
        for n_sensors in n_sensors_list:
            results_list = []
            for _ in range(m):
                sampler = generate_sensors(n_sensors, 0)
                estimator = KSVDRainReconstricution(sampler, ksvd_learner, mean_vector)
                final_res = validation(estimator)
                results_list.append(final_res)
            print("a")

    if False:
        results_list = []
        n_sensors_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]
        for n_sensors in n_sensors_list:
            sampler = generate_sensors(n_sensors, 0)
            estimator = KSVDRainReconstricution(sampler, ksvd_learner, mean_vector)
            final_res = validation(estimator)
            results_list.append(final_res)

        plt.plot(n_sensors_list, results_list, label="Point Sensor")
        plt.grid()
        plt.xlabel("Number of Sensors")
        plt.ylabel("MSE")
        # plt.legend()
        plt.tight_layout()
        plt.savefig("n_sensors.svg")
        # plt.show()
    if False:
        # sampler = generate_sensors(0, 30)
        # estimator = KSVDRainReconstricution(sampler, ksvd_learner, mean_vector)
        # final_res = validation(estimator, "line", plot_maps=True)
        # print(final_res)
        sampler = generate_sensors(30, 0)
        estimator = KSVDRainReconstricution(sampler, ksvd_learner, mean_vector)
        final_res = validation(estimator, "point", plot_maps=True)
        print(final_res)
