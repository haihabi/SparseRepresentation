import matplotlib.pyplot as plt

import dataset
import numpy as np
import dictionary_learning

data_cml = True
patch_size = -1
filter_data = False
max_data_size = np.inf
validation_size = 512


class Sampler():
    def __init__(self, point_sensors, in_shape):
        self.point_sensors = point_sensors

        x_mid = self.point_sensors[:, 0] * 15
        y_mid = self.point_sensors[:, 1] * 15
        self.x_c = np.ceil(x_mid).astype("int")
        self.x_f = np.floor(x_mid).astype("int")
        self.y_c = np.ceil(y_mid).astype("int")
        self.y_f = np.floor(y_mid).astype("int")

        self.s_x = x_mid / self.x_c
        self.s_y = y_mid / self.y_c

        cc_index_flat = np.ravel_multi_index(np.stack([self.x_c, self.y_c]), in_shape)
        cf_index_flat = np.ravel_multi_index(np.stack([self.x_c, self.y_f]), in_shape)
        fc_index_flat = np.ravel_multi_index(np.stack([self.x_f, self.y_c]), in_shape)
        ff_index_flat = np.ravel_multi_index(np.stack([self.x_f, self.y_f]), in_shape)

        h_matrix = np.zeros([self.point_sensors.shape[0], np.prod(in_shape)])
        for i in range(self.point_sensors.shape[0]):
            h_matrix[i, cc_index_flat[i]] = self.s_x[i] * self.s_y[i]
            h_matrix[i, cf_index_flat[i]] = self.s_x[i] * (1 - self.s_y[i])
            h_matrix[i, fc_index_flat[i]] = (1 - self.s_x[i]) * self.s_y[i]
            h_matrix[i, ff_index_flat[i]] = (1 - self.s_x[i]) * (1 - self.s_y[i])
        self.h_matrix = h_matrix
        # print("a")

    def sample(self, in_field):
        return self.h_matrix @ in_field.flatten()


def generate_sensors(n_p, n_l, min_point=-5, max_point=5):
    s = np.random.rand(n_p, 2)
    return Sampler(s, (16, 16))


# def reconstruction_matching_pursuit(in_dictionary, in_y_matrix, in_k_sparse, max_iter=100) -> np.ndarray:
#     # analyze shape of Y
#     if len(in_y_matrix.shape) == 1:
#         data = np.array([in_y_matrix])
#     elif len(in_y_matrix.shape) == 2:
#         data = in_y_matrix
#     else:
#         raise ValueError("Input must be a vector or a matrix.")
#     # analyze dimensions
#     if not in_dictionary.shape[0] == in_y_matrix.shape[0]:
#         raise ValueError("Dimension mismatch: %s != %s" % (in_dictionary.shape[0], in_y_matrix.shape[0]))
#
#     alphas = []
#     for y in data.T:
#         # temporary values
#         coeffs = np.zeros(in_dictionary.shape[1])
#         residual = y
#
#         # iterate
#         i = 0
#         if max_iter:
#             m = max_iter
#         else:
#             m = np.inf
#
#         finished = False
#
#         while not finished:
#             if i >= m:
#                 break
#             inner = np.dot(in_dictionary.T, residual)
#             gamma = int(np.argmax(np.abs(inner)))
#             alpha = inner[gamma]
#             residual = residual - alpha * in_dictionary[:, gamma]
#             if np.isclose(alpha, 0):
#                 break
#             coeffs[gamma] += alpha
#             i += 1
#             finished = np.count_nonzero(coeffs) >= in_k_sparse
#
#         alphas.append(coeffs)
#     return np.transpose(alphas)


data_training, data_validation, mean_vector = dataset.get_dataset(data_cml, patch_size, filter_data, validation_size,
                                                                  max_data_size)

sampler = generate_sensors(30, 0)

ksvd_config = dictionary_learning.KSVDConfig(200, 30, 512)
ksvd_learner = dictionary_learning.KSVD(ksvd_config)
ksvd_learner.batch_training(data_training, batch_size=256)



#######################
# Modified MP
######################
d_hat = sampler.h_matrix @ ksvd_learner.dictionary_best

rain_field = np.reshape(data_validation[0, :], [16, 16])
y = sampler.sample(rain_field)
x = dictionary_learning.sparse_recovery.matching_pursuit(d_hat, np.expand_dims(y, axis=1), 30)
r_hat = ksvd_learner.dictionary_best @ x + mean_vector.T

r_hat = r_hat.reshape(rain_field.shape)
from matplotlib import pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(rain_field)
plt.subplot(1, 2, 2)
plt.imshow(r_hat)
plt.show()
print("a")
