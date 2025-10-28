import glob
import os
import cv2
import numpy as np
from .utils import read_mat
from .abc import HSSRDS, HSSRTransform
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Iterable
from multiprocessing import shared_memory
from .utils import remove_shm_from_resource_tracker
import multiprocessing as mp
import tqdm


class ARADDatasetRe(HSSRDS):
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transforms: HSSRTransform = None,
        preload: bool = True,
        rank: int = 0,
        world_size: int = 0
    ) -> None:
        super().__init__()
        self.bands = 31
        self.rank = rank
        self.root_dir = root_dir
        set_name = "Train" if is_train else "Valid"
        ds_name = 'ARAD_RE'
        self.mat_dir = os.path.join(root_dir, set_name + "_spectral")
        self._mat_file_paths = sorted(glob.glob(os.path.join(self.mat_dir, "*.mat")))
        self.rgb_dir = os.path.join(root_dir, set_name + "_RGB_re")
        self._rgb_file_paths = sorted(glob.glob(os.path.join(self.rgb_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        assert len(self._mat_file_paths) == len(self._rgb_file_paths)
        # print(self._mat_file_paths)
        self.transforms = transforms
        self._use_shared_mm = False
        H = 482
        W = 512
        if world_size > 0:
            self._use_shared_mm = True
            remove_shm_from_resource_tracker()
            creat_mem = True # if rank == 0 else False
            size_hsi = len(self._mat_file_paths) * 4 * 31 * H * W # if rank == 0 else 0
            size_rgb = len(self._mat_file_paths) * 4 * 3 * H * W # if rank == 0 else 0
            size_loaded = len(self._mat_file_paths) * 4
            try:
                self._mat_mm = shared_memory.SharedMemory('%s_hsi_%s'%(ds_name, set_name), create=creat_mem, size=size_hsi)
                self._rgb_mm = shared_memory.SharedMemory('%s_rgb_%s'%(ds_name, set_name), create=creat_mem, size=size_rgb)
                self._loaded_mm = shared_memory.SharedMemory('%s_loaded_%s'%(ds_name, set_name), create=creat_mem, size=size_loaded)
                self._creat_mem = True
            except FileExistsError:
                self._mat_mm = shared_memory.SharedMemory('%s_hsi_%s'%(ds_name, set_name))
                self._rgb_mm = shared_memory.SharedMemory('%s_rgb_%s'%(ds_name, set_name))
                self._loaded_mm = shared_memory.SharedMemory('%s_loaded_%s'%(ds_name, set_name))
                self._creat_mem = False

            self._mat_np = np.ndarray((len(self._mat_file_paths), 31, H, W), dtype=np.float32, buffer=self._mat_mm.buf)
            self._rgb_np = np.ndarray((len(self._mat_file_paths), 3, H, W), dtype=np.float32, buffer=self._rgb_mm.buf)
            self._loaded_np = np.ndarray((len(self._mat_file_paths),), dtype=np.uint32, buffer=self._loaded_mm.buf)
            if self._creat_mem:
                self._loaded_np.fill(0)
        else:
            self._mat_np = np.empty((len(self._mat_file_paths), 31, H, W), dtype=np.float32)
            self._rgb_np = np.empty((len(self._mat_file_paths), 3, H, W), dtype=np.float32)
            self._loaded_np = np.zeros((len(self._mat_file_paths),), dtype=np.uint32)

        # self._preloaded = [False for _ in range(len(self._mat_file_paths))]
        self._save2mem = preload
        

    def _load_data(self, index):
        mat = read_mat(self._mat_file_paths[index], "cube")
        mat = np.transpose(mat, (0, 2, 1))
        rgb = cv2.imread(self._rgb_file_paths[index])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        if self._save2mem:
            # print('saving ', index, mat.shape, rgb.shape)
            self._mat_np[index] = mat
            self._rgb_np[index] = rgb
            self._loaded_np[index] = 0x0001
        return mat, rgb
    
    def set_use_cache(self, use_cache:bool = True):
        self._preloaded = use_cache

    def __getitem__(self, index) -> Tuple[np.ndarray]:
        if self._loaded_np[index] > 0:
            # assert isinstance(self._mat_list, list), "Mat list has to be list"
            # assert isinstance(self._rgb_list, list), "Img list has to be list"
            # mat = self._mat_list[index]
            # rgb = self._rgb_list[index]
            mat = self._mat_np[index].copy()
            rgb = self._rgb_np[index].copy()
        else:
            mat, rgb = self._load_data(index)
            
        if self.transforms is not None:
            mat, rgb = self.transforms.transform(mat, rgb)

        return mat, rgb

    def __len__(self):
        return len(self._mat_file_paths)
    
    def load_all(self):
        for i in tqdm.tqdm(range(len(self._mat_file_paths))):
            self._load_data(i)              
    
    def close_sharedmm(self, unlink:bool = False):
        del self._mat_np, self._rgb_np, self._loaded_np
        if self._use_shared_mm:
            self._mat_mm.close()
            self._rgb_mm.close()     
            self._loaded_mm.close()
            if self._creat_mem or unlink:
                self._mat_mm.unlink()
                self._rgb_mm.unlink()
                self._loaded_mm.unlink()

def preload_all_into_sharedmm():
    base_dir = '/mnt/data/ARAD/'
    ds_train = ARADDatasetRe(base_dir,True,None,True,0,1)
    while not ds_train._creat_mem: #"Shared memory has been created. Please check"
        ds_train.close_sharedmm(unlink=True)
        del ds_train
        ds_train = ARADDatasetRe(base_dir,True,None,True,0,1)
    print('Creating the shared memory.')
    print('loading Training dataset')
    ds_train.load_all()
    ds_valid = ARADDatasetRe(base_dir,False,None,True,0,1)
    while not ds_valid._creat_mem: #"Shared memory has been created. Please check"
        ds_valid.close_sharedmm(unlink=True)
        del ds_valid
        ds_valid = ARADDatasetRe(base_dir,False,None,True,0,1)
    print('loading Valid dataset')
    ds_valid.load_all()
    print("If no longer need the data, simply enter \"exit\"")
    while True:
        x = input()
        if x == "exit":
            break

    ds_train.close_sharedmm()
    ds_valid.close_sharedmm()

if __name__ == "__main__":
    import time

    # ds = ARADDatasetRe("/mnt/data/ARAD/", True, None, preload=True)
    # random_permute = np.random.permutation(len(ds))
    # for idx, randidx in enumerate(random_permute):    
    #     mat, rgb = ds[randidx]
    #     print(idx, randidx, mat.shape, rgb.shape)
    #     time.sleep(0.01)
    # del ds
    preload_all_into_sharedmm()
