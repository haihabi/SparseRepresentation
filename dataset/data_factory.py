import netCDF4
import numpy as np
from dataset.data_downloader import download_open_mrg, transform_open_mrg
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from skimage import data


def get_dataset(data_cml, patch_size, filter_data, validation_size, max_data_size):
    if data_cml:
        data_folder = r"data\\"
        download_open_mrg(local_path=r"data\\")
        file_location = data_folder + "OpenMRG.zip"
        transform_open_mrg(file_location, data_folder)
        file_location = data_folder + f"radar\\radar.nc"
        radar_dataset = netCDF4.Dataset(file_location)
        data_list = []
        for i in tqdm(range(radar_dataset.variables['data'].shape[0])):
            x = np.asarray(radar_dataset.variables['data'][i, :, :])
            if np.all(x != 255):
                radar_rain_tensor = np.power(10, (x / 15)) * np.power(1 / 200, 1 / 1.5)
                if np.sum(radar_rain_tensor) > 15:
                    if patch_size > 0:
                        patch_tensors = extract_patches_2d(radar_rain_tensor, (patch_size, patch_size),
                                                           max_patches=2048)
                    else:
                        patch_tensors = np.expand_dims(radar_rain_tensor[16:-16, 10:-11], axis=0)
                    data_list.append(patch_tensors)

        data_raw = np.concatenate(data_list, axis=0)
        if patch_size <= 0:
            patch_size = 16
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
    return data_training, data_validation, mean_vector
