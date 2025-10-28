import glob
import math
import h5py
import scipy.io as sio
import numpy as np
from scipy import integrate
import scipy.interpolate as si
import pickle

import tqdm
import os
import cv2

import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from typing import Iterator, Optional, List
import pandas as pd

from multiprocessing import Process, resource_tracker
from multiprocessing.shared_memory import SharedMemory


class DSBalanceSampler(Sampler):
    def __init__(self, datasets) -> None:
        weights = []
        len_cum = 0
        for ds in datasets:
            weights.extend([len(ds)] * len(ds))
            len_cum += len(ds)
        self.weights = len_cum - torch.as_tensor(weights, dtype=torch.double)
        # self.num_samples = len_cum
        self.replacement = True
        self.num_samples = len_cum
    
    def __iter__(self) -> Iterator:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=None
        )
        indices = rand_tensor.tolist()
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class DistDSBalanceSampler(Sampler):
    def __init__(
        self,
        datasets,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        weights = []
        len_cum = 0
        for ds in datasets:
            weights.extend([len(ds)] * len(ds))
            len_cum += len(ds)
        self.weights = len_cum - torch.as_tensor(weights, dtype=torch.double)
        self.replacement = True

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len_cum % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len_cum - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len_cum / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rand_tensor = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=g
        )
        indices = rand_tensor.tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

def read_mat(mat_file_path: str, name: str = None):
    """_summary_

    Args:
        mat_file_path (_type_): _description_
    """

    def _read_mat_sio():
        mat = sio.loadmat(mat_file_path)
        return mat

    def _read_mat_h5():
        f = h5py.File(mat_file_path, "r")
        if name is not None:
            data = f.get(name=name)
            mat = np.array(data)
        else:
            mat = dict(f.items())
            for key, value in mat.items():
                mat[key] = np.array(value)

        return mat

    try:
        mat = _read_mat_sio()
    except:
        mat = _read_mat_h5()
    return mat


def read_response_dso(file_path: str):
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    srfs = np.array(loaded["SRF"])
    specs = np.array(loaded["dispwl"])
    b, g, r = srfs
    return b, g, r, specs


class RenderMS(object):
    def __init__(self, cie_file_path: str, analog_gain:List[float]=[2.01122684, 1.0, 1.697217033]) -> None:
        self.cie_file_path = cie_file_path
        self.spec = []
        self.x = []
        self.y = []
        self.z = []
        if cie_file_path.endswith("csv"):
            self._read_cie_file_csv()
        elif cie_file_path.endswith("pkl"):
            self._read_cie_file_pkl()
        else:
            raise NotImplementedError
        # self.analog_gain = np.array([2.2933984, 1, 1.62308182]).reshape(1,1,3)
        self.analog_gain = np.array(analog_gain).reshape(1, 1, 3)

    def _read_cie_file_csv(self):
        with open(self.cie_file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            spec, x, y, z = line.split(",")
            self.spec.append(float(spec.strip("\r").strip("\n")))
            self.x.append(float(x.strip("\r").strip("\n")))
            self.y.append(float(y.strip("\r").strip("\n")))
            self.z.append(float(z.strip("\r").strip("\n")))
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.spec = np.array(self.spec)
        # df = pd.read_csv(self.cie_file_path)
        # self.x = df['R'].to_numpy()
        # self.y = (df['G1'].to_numpy()+df['G2'].to_numpy()) / 2.0
        # self.z = df['B'].to_numpy()
        # self.spec = df['Wavelength[nm]'].to_numpy()
        # print(self.spec, self.x, self.y, self.z)
        

    def _read_cie_file_pkl(self):
        b, g, r, specs = read_response_dso(self.cie_file_path)
        self.x = r
        self.y = g
        self.z = b
        self.spec = specs

    def render_ms(self, ms_response: np.ndarray, ms_spec: np.ndarray):
        ms_response = ms_response.astype(np.float32)
        rgb_result = np.zeros(
            (
                ms_response.shape[0],
                ms_response.shape[1],
                3,
            ),
            dtype=np.float32,
        )

        for i, color_response in enumerate([self.x, self.y, self.z]):
            inter_f = si.interp1d(self.spec, color_response, bounds_error=False, fill_value=0.0)
            response_index = inter_f(ms_spec)
            # rgb_result[..., i] = np.sum(ms_response * response_index[None, None, :], axis=2)#  / np.sum(response_index)
            # rgb_result[..., i] = integrate.trapezoid(
            #     ms_response * response_index[None, None, :], ms_spec, axis=2) / integrate.trapezoid(response_index, ms_spec)
            # print('%d: '%i, integrate.trapezoid(response_index, ms_spec))
            # if i == 0:
            #     # K = 1.0 / integrate.trapezoid(response_index, ms_spec)
            #     K = 1.0 / 5e6
            rgb_result[..., i] = integrate.trapezoid(
                ms_response * response_index[None, None, :], ms_spec, axis=2
            )
        rgb_result = rgb_result * self.analog_gain
        return rgb_result
    
    def interpolate_hsi(self, ms_response: np.ndarray, ms_spec: np.ndarray, target_spec: np.ndarray):
        inter_f = si.interp1d(ms_spec, ms_response, bounds_error=False, fill_value=0.0, axis=2)
        radiance_interpolated = inter_f(target_spec)
        return radiance_interpolated


def render_arad_img():
    mat_dir = "/mnt/data/ARAD/Train_spectral/"
    mat_dir_valid = "/mnt/data/ARAD/Valid_spectral/"
    out_dir = "AAA_RESPONSE4"
    out_dir_valid = "AAA_RESPONSE4_valid"
    # file_path = "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/a.pkl"
    file_path = "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/RGB_Camera_QE.csv"
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    RM = RenderMS(file_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid, exist_ok=True)
    specs = np.linspace(400, 700, 31)

    dataset_sum = [0.0, 0.0, 0.0]  # in RGB direction
    dataset_sum_sq = [0.0, 0.0, 0.0]  # in RGB direction
    all_rendered_data = {}
    count = 0
    for idx, mat_file in enumerate(os.listdir(mat_dir)):
        if not mat_file.endswith("mat"):
            continue
        
        mat = read_mat(os.path.join(mat_dir, mat_file), "cube")
        mat = np.transpose(mat, (2, 1, 0))
        
        rendered = RM.render_ms(mat, specs)
        h, w, _ = rendered.shape
        count += h * w

        # for c_idx in range(3):
        #     dataset_sum[c_idx] += np.sum(rendered[:,:,c_idx])
        #     dataset_sum_sq[c_idx] += np.sum(rendered[:,:,c_idx] ** 2)
        dataset_sum_old = dataset_sum.copy()
        dataset_sum = [
            sum_c + np.sum(rendered[:,:,c_idx])
            for c_idx, sum_c in enumerate(dataset_sum_old)
        ]
        dataset_sum_sq_news = [
            sum_c + np.sum(rendered[:,:,c_idx] ** 2)
            for c_idx, sum_c in enumerate(dataset_sum_sq)
        ]
        dataset_sum_sq = dataset_sum_sq_news
        save_name = os.path.join(out_dir, mat_file.replace(".mat", ".png"))
        all_rendered_data[save_name] = rendered

    dataset_mean = np.array(dataset_sum) / count
    dataset_var = np.array(dataset_sum_sq) / count - dataset_mean ** 2
    dataset_std = np.sqrt(dataset_var)
    print("Dataset_mean: ", dataset_mean)
    print("Dataset_std: ", dataset_std)
    np.save(os.path.join('/mnt/data/ARAD', 'rgb_mean_ntire.npy'), dataset_mean)
    np.save(os.path.join('/mnt/data/ARAD', 'rgb_std_ntire.npy'), dataset_std)

    dataset_mean = dataset_mean.reshape(1,1,3)
    dataset_std = dataset_std.reshape(1,1,3)
    print("Saving imgs")
    
    for save_name, rendered in tqdm.tqdm(all_rendered_data.items()):
        img = np.copy(rendered)
        img = (img - dataset_mean) / dataset_std
        img = (img * target_std) + target_mean
        img = np.clip(img, 0, 1.0)
        img = np.round(img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, img)
    
    print("Processing Validation images")
    # validation set
    for idx, mat_file in tqdm.tqdm(enumerate(os.listdir(mat_dir_valid))):
        if not mat_file.endswith("mat"):
            continue
        
        mat = read_mat(os.path.join(mat_dir_valid, mat_file), "cube")
        mat = np.transpose(mat, (2, 1, 0))
        
        rendered = RM.render_ms(mat, specs)
        img = np.copy(rendered)
        img = (img - dataset_mean) / dataset_std
        img = (img * target_std) + target_mean
        img = np.clip(img, 0, 1.0)
        img = np.round(img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_name = os.path.join(out_dir_valid, mat_file.replace(".mat", ".png"))
        cv2.imwrite(save_name, img)



def render_icvl_img():
    mat_dir = "/mnt/data/ICVL/original/"
    mat_dir_valid = "/mnt/data/ARAD/Valid_spectral/"
    out_dir = "/mnt/data/ICVL/rendered_ntire"
    # out_dir_valid = "AAA_RESPONSE3_valid"
    # file_path = "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/a.pkl"
    file_path = "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/RGB_Camera_QE.csv"
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    RM = RenderMS(file_path)

    os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(out_dir_valid, exist_ok=True)
    specs = np.linspace(400, 700, 31)

    dataset_sum = [0.0, 0.0, 0.0]  # in RGB direction
    dataset_sum_sq = [0.0, 0.0, 0.0]  # in RGB direction
    all_rendered_data = {}
    count = 0
    
    dataset_mean = np.load(os.path.join('/mnt/data/ARAD', 'rgb_mean_ntire.npy'))
    dataset_std = np.load(os.path.join('/mnt/data/ARAD', 'rgb_std_ntire.npy'))

    for idx, mat_file_path in enumerate(glob.glob(os.path.join(mat_dir, "*.mat"))):
        if mat_file_path.endswith('_31c.mat'):
            continue
        mat = read_mat(mat_file_path)
        mat_file_name = os.path.basename(mat_file_path)
        if "__header__" in mat.keys():
            mat = mat["ref"].astype(np.float32) / 4095.0
        else:
            mat = np.transpose(mat["rad"], (1, 2, 0))[::-1, :, :]
            mat = mat.astype(np.float32) / 4095.0
        print(mat_file_path, '   ', np.max(mat), '   ', np.min(mat))
        rendered = RM.render_ms(mat, specs)
        h, w, _ = rendered.shape
        count += h * w

        for c_idx in range(3):
            dataset_sum[c_idx] += np.sum(rendered[:,:,c_idx])
            dataset_sum_sq[c_idx] += np.sum(rendered[:,:,c_idx] ** 2)
        # dataset_sum_old = dataset_sum.copy()
        # dataset_sum = [
        #     sum_c + np.sum(rendered[:,:,c_idx])
        #     for c_idx, sum_c in enumerate(dataset_sum_old)
        # ]
        # dataset_sum_sq_news = [
        #     sum_c + np.sum(rendered[:,:,c_idx] ** 2)
        #     for c_idx, sum_c in enumerate(dataset_sum_sq)
        # ]
        # dataset_sum_sq = dataset_sum_sq_news
        save_name = os.path.join(out_dir, mat_file_name.replace(".mat", ".png"))
        all_rendered_data[save_name] = rendered

    # dataset_mean = np.array(dataset_sum) / count
    # dataset_var = np.array(dataset_sum_sq) / count - dataset_mean ** 2
    # dataset_std = np.sqrt(dataset_var)
    # print("Dataset_mean: ", dataset_mean)
    # print("Dataset_std: ", dataset_std)

    dataset_mean = dataset_mean.reshape(1,1,3)
    dataset_std = dataset_std.reshape(1,1,3)
    print("Saving imgs")
    
    for save_name, rendered in tqdm.tqdm(all_rendered_data.items()):
        img = np.copy(rendered)
        img = (img - dataset_mean) / dataset_std
        img = (img * target_std) + target_mean
        img = np.clip(img, 0, 1.0)
        img = np.round(img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, img)




if __name__ == "__main__":
    
    # render_arad_img()
    # render_icvl_img()
    # file_path = "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/a.pkl"
    # RM = RenderMS(file_path)
    # mat_dir = "/mnt/data/ARAD/Train_spectral/"
    # out_dir = "AAA_RESPONSE2"
    # os.makedirs(out_dir, exist_ok=True)
    # specs = np.linspace(400, 700, 31)
    # for mat_file in os.listdir(mat_dir):
    #     if not mat_file.endswith("mat"):
    #         continue
    #     mat = read_mat(os.path.join(mat_dir, mat_file), "cube")
    #     mat = np.transpose(mat, (2, 1, 0))
    #     rendered = RM.render_ms(mat, specs)
    #     rendered = np.round(rendered * 255).clip(0, 255).astype(np.uint8)
    #     rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(out_dir, mat_file.replace(".mat", ".png")), rendered)
    file_path = "/mnt/data/ICVL/original/nachal_0823-1217.mat"
    mat = read_mat(file_path)
    if "__header__" in mat.keys():
        mat = mat["ref"].astype(np.float32) / 4095.0
    else:
        mat = np.transpose(mat["rad"], (1, 2, 0))[::-1, :, :]
        mat = mat.astype(np.float32) / 4095.0
    print(mat)
    mat = cv2.resize(mat, None, fx=0.3, fy=0.3)
    mat = np.transpose(mat, (2, 0, 1))
    print(mat.shape)
    np.save('1217.npy', mat)