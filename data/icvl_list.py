import enum
import glob
import os
import tqdm
from multiprocessing import shared_memory
from .abc import HSSRDS, HSSRTransform
from .utils import read_mat, RenderMS
import numpy as np
import cv2
import patchify as ptf

# the keys are the finetuning training set,
# the values are lists of corresponding test data
ICVL_FT_DATA = {
    "4cam_0411-1640-1": "4cam_0411-1648",
    "bgu_0403-1444": "bgu_0403-1459",
    "BGU_0403-1419-1": "bgu_0403-1439",
    "bgu_0403-1523": "bgu_0403-1511",
    "BGU_0522-1136": "BGU_0522-1127",
    "BGU_0522-1201": ["BGU_0522-1203", "BGU_0522-1211", "BGU_0522-1216"],
    "bulb_0822-0909": ["bulb_0822-0903", "objects_0924-1650"],
    "bguCAMP_0514-1712": "bguCAMP_0514-1711",
    "bguCAMP_0514-1718": "bguCAMP_0514-1723",
    "eve_0331-1647": ["eve_0331-1646", "eve_0331-1656", "eve_0331-1657"],
    "eve_0331-1702": "eve_0331-1705",
    "gavyam_0823-0945": "gavyam_0823-0950-1",
    "hill_0325-1219": ["hill_0325-1228", "hill_0325-1235", "hill_0325-1242"],
    "IDS_COLORCHECK_1020-1215-1": "IDS_COLORCHECK_1020-1223",
    "Labtest_0910-1506": [
        "Labtest_0910-1509",
        "Labtest_0910-1511",
        "Labtest_0910-1513",
    ],
    "lehavim_0910-1605": ["lehavim_0910-1607", "lehavim_0910-1610"],
    "Lehavim_0910-1622": ["Lehavim_0910-1626", "Lehavim_0910-1627"],
    "Lehavim_0910-1629": "Lehavim_0910-1630",
    "Lehavim_0910-1716": "Lehavim_0910-1717",
    "Lehavim_0910-1725": "Lehavim_0910-1718",
    "nachal_0823-1040": ["nachal_0823-1038", "nachal_0823-1047"],
    "nachal_0823-1110": "nachal_0823-1121",
    "nachal_0823-1117": "nachal_0823-1118",
    "nachal_0823-1132": "nachal_0823-1144",
    "nachal_0823-1145": "nachal_0823-1147",
    "nachal_0823-1152": "nachal_0823-1149",
    "nachal_0823-1213": "nachal_0823-1214",
    "nachal_0823-1217": ["nachal_0823-1222", "nachal_0823-1223"],
    "pepper_0503-1228": [
        "pepper_0503-1229",
        "pepper_0503-1236",
        "peppers_0503-1308",
        "peppers_0503-1330",
        "peppers_0503-1332",
    ],
    "peppers_0503-1315": "peppers_0503-1315",
    "plt_0411-1116": ["plt_0411-1037", "plt_0411-1046"],
    "plt_0411-1155": "plt_0411-1200-1",
    "plt_0411-1210": ["plt_0411-1207", "plt_0411-1211"],
    "prk_0328-1025": "prk_0328-1031",
    "prk_0328-1037": "prk_0328-1045",
}

for train, test in ICVL_FT_DATA.items():
    ICVL_FT_DATA[train] = test if isinstance(test, str) else test[0]

def _creat_inv_dict():
    inv_dict = {}
    for train_name, test_names in ICVL_FT_DATA.items():
        if isinstance(test_names, str):
            inv_dict[test_names] = train_name
        elif isinstance(test_names, list):
            for one_name in test_names:
                inv_dict[one_name] = train_name
        else:
            raise RuntimeError("ICVL_FT_DATA dict def error!")
    return inv_dict


def _get_ft_test_set():
    test_train_dict = _creat_inv_dict()
    return list(test_train_dict.keys())


def _get_all_train_set():
    return list(ICVL_FT_DATA.keys())


def _get_train_for_ft(huge_test_error_names):
    assert isinstance(huge_test_error_names, list)
    if isinstance(huge_test_error_names[0], int):
        test_names = np.array(_get_ft_test_set())
        huge_test_error_names = test_names[huge_test_error_names].tolist()
    train_samples = list()
    inv_dict = _creat_inv_dict()
    test_samples = list(inv_dict.keys())
    for name in huge_test_error_names:
        if isinstance(name, int):
            name = test_samples[name]
        train_samples.append(inv_dict[name])
    train_samples = list(set(train_samples))
    return train_samples


class ICVLFTDataset(HSSRDS):
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        transforms: HSSRTransform = None,
        preload: bool = True,
        rank: int = 0,
        world_size: int = 0,
        test_set: list = None,
        random_ratio: float = 0.3,
        scale: float = 0.3
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self._transforms = transforms
        if is_train:
            if test_set is None:
                mat_name_list = _get_all_train_set()
                train_nums = int(len(mat_name_list) * random_ratio)
                self._mat_name_list = np.random.permutation(mat_name_list)[:train_nums]
            else:
                mat_name_list = _get_train_for_ft(test_set)
                self._mat_name_list = mat_name_list
        else:
            mat_name_list = _get_ft_test_set()
            self._mat_name_list = mat_name_list
        print(self._mat_name_list)
        self._mat_file_list = [
            os.path.join(self.data_root, "original", name + ".mat")
            for name in self._mat_name_list
        ]
        self._rgb_file_list = [
            os.path.join(self.data_root, "rendered_dso2", name + ".png")
            for name in self._mat_name_list
        ]
        self.is_train = is_train
        self.rank = rank
        self.scale = scale

        H = 1392
        W = 1300
        if world_size > 0 and is_train:
            self._distributed = True
            set_name = "Train" if is_train else "Valid"
            creat_mem = True  # if rank == 0 else False
            size_hsi = len(self._mat_file_list) * 4 * 31 * H * W  # if rank == 0 else 0
            size_rgb = len(self._mat_file_list) * 4 * 3 * H * W  # if rank == 0 else 0
            try:
                self._mat_mm = shared_memory.SharedMemory(
                    "icvl_hsi_%s" % set_name, create=creat_mem, size=size_hsi
                )
                self._rgb_mm = shared_memory.SharedMemory(
                    "icvl_rgb_%s" % set_name, create=creat_mem, size=size_rgb
                )
                self._loaded_mm = shared_memory.SharedMemory(
                    "icvl_loaded_%s" % set_name,
                    create=creat_mem,
                    size=len(self._mat_file_list) * 4 * 3,
                )
            except FileExistsError:
                self._mat_mm = shared_memory.SharedMemory("icvl_hsi_%s" % set_name)
                self._rgb_mm = shared_memory.SharedMemory("icvl_rgb_%s" % set_name)
                self._loaded_mm = shared_memory.SharedMemory(
                    "icvl_loaded_%s" % set_name
                )
            self._mat_np = np.ndarray(
                (len(self._mat_file_list), 31, H, W),
                dtype=np.float32,
                buffer=self._mat_mm.buf,
            )
            self._rgb_np = np.ndarray(
                (len(self._mat_file_list), 3, H, W),
                dtype=np.float32,
                buffer=self._rgb_mm.buf,
            )
            self._loaded_np = np.ndarray((len(self._mat_file_list), 3), dtype=np.uint32, buffer=self._loaded_mm.buf)
            if self.rank == 0:
                self._loaded_np[:] = 0

        else:
            self._mat_np = np.empty(
                (len(self._mat_file_list), 31, H, W), dtype=np.float32
            )
            self._rgb_np = np.empty(
                (len(self._mat_file_list), 3, H, W), dtype=np.float32
            )
            self._loaded_np = np.zeros((len(self._mat_file_list), 3), dtype=np.uint32)
            # self._max_h = np.zeros((len(self._mat_file_list),), dtype=np.int32)
            # self._max_w = np.zeros((len(self._mat_file_list),), dtype=np.int32)

        # self._preloaded = [False for _ in range(len(self._mat_file_list))]
        self._save2mem = preload

    def _check_preloaded(self, index):
        return self._loaded_np[index, 0] > 0

    def __getitem__(self, index) -> any:
        if self._check_preloaded(index):
            hyper_spectral = self._mat_np[index]
            rgb = self._rgb_np[index]
            h, w = self._loaded_np[index, 1], self._loaded_np[index, 2]
            hyper_spectral = hyper_spectral[:, :h, :w]
            rgb = rgb[:, :h, :w]
        else:
            hyper_spectral, rgb = self._load_data(index)
            h, w = rgb.shape[1], rgb.shape[2]
            # print(h, w)
            # print(self._max_h, self._max_w)
            if self._save2mem:
                self._mat_np[index, :, :h, :w] = hyper_spectral
                self._rgb_np[index, :, :h, :w] = rgb
                # self._max_h[index] = h
                # self._max_w[index] = w
                self._loaded_np[index, 0] = 1
                self._loaded_np[index, 1] = h
                self._loaded_np[index, 2] = w
        # print(hyper_spectral.shape)
        if self._transforms is not None:
            hyper_spectral, rgb = self._transforms.transform(hyper_spectral, rgb)

        return hyper_spectral, rgb

    def _read_mat_icvl(self, mat_file_path):
        mat = read_mat(mat_file_path)
        if "__header__" in mat.keys():
            mat = mat["ref"].astype(np.float32) / 4095.0
        else:
            mat = np.transpose(mat["rad"], (1, 2, 0))[::-1, :, :]
            mat = mat.astype(np.float32) / 4095.0
        return mat

    def _load_data(self, index):
        mat = self._read_mat_icvl(self._mat_file_list[index])
        if not self.scale == 1.0:
            mat = cv2.resize(mat, None, fx=self.scale, fy=self.scale)
        mat = np.transpose(mat, (2, 0, 1))
        rgb = cv2.imread(self._rgb_file_list[index])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if not self.scale == 1.0:
            rgb = cv2.resize(rgb, None, fx=self.scale, fy=self.scale)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return mat, rgb

    def __len__(self):
        if hasattr(self, "_mat_list"):
            return len(self._mat_list)
        else:
            return len(self._mat_file_list)


def crop_icvl_dataset(
    data_root: str, save_root: str, is_train: bool, crop_size: int = 256
):
    for mat_file in sorted(glob.glob(os.path.join(data_root, "*.mat"))):
        mat = read_mat(mat_file)


def xyz2bgr(xyz: np.ndarray):
    x, y, z = np.split(xyz, 3, 2)
    r = 2.3646 * x - 0.8965 * y - 0.4681 * z
    g = -0.5152 * x + 1.4264 * y + 0.0888 * z
    b = 0.0052 * x - 0.0144 * y + 1.0092 * z
    bgr = np.concatenate([b, g, r], axis=2)
    print(bgr.shape)
    return bgr


def visualize_icvl_dataset(data_root: str, save_root: str):
    os.makedirs(save_root, exist_ok=True)

    render = RenderMS(
        # "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/lin2012xyz10e_5_7sf.csv"
        "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/ciexyz64_1.csv"
    )
    specs = np.linspace(400, 700, 31)
    f2uint8 = lambda x: np.clip(np.round(x * 255), 0, 255).astype(np.uint8)
    for mat_file in sorted(glob.glob(os.path.join(data_root, "*.mat"))):
        mat = read_mat(mat_file)
        if "__header__" in mat.keys():
            print(mat_file, "Into __header__")
            # rgb_ori = mat["rgb"]
            # rgb_ori = (rgb_ori - rgb_ori.min()) / (rgb_ori.max() - rgb_ori.min())
            # rgb_ori = rgb_ori.astype(np.float32)
            mat = mat["ref"] / 4096
        else:
            # continue
            # rgb_ori = np.transpose(mat["rgb"], (2, 1, 0))
            # rgb_ori = f2uint8(rgb_ori)
            mat = np.transpose(mat["rad"], (1, 2, 0))[::-1, :, :]
            # mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))  # 4096
            mat = mat / 4095
        print(np.histogram(mat))
        rgb = render.render_ms(mat, specs)
        # rgb = np.stack([mat[:,:,3], mat[:,:,8], mat[:,:,12]], axis=-1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_XYZ2BGR)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        # rgb = xyz2bgr(rgb)
        print(np.histogram(rgb))
        # out_path_ori = os.path.join(
        # save_root, os.path.basename(mat_file).replace(".mat", "_ori.png")
        # )
        out_path_rer = os.path.join(
            save_root, os.path.basename(mat_file).replace(".mat", "_rer.png")
        )

        rgb = f2uint8(rgb)
        # cv2.imwrite(out_path_ori, cv2.cvtColor(rgb_ori, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_path_rer, rgb)


if __name__ == "__main__":
    # visualize_icvl_dataset("/mnt/data/ICVL/original/", "/mnt/data/ICVL/rgb_rendered/")
    # ds = ICVLDataset(
    #     "/mnt/data/ICVL/original/",
    #     "/home/rudolfmaxx/projects/MulitpleHSSR/data/cie_files/ciexyz64_1.csv",
    #     False,
    #     None,
    #     True,
    # )
    # count = 0
    # for mat, rgb in ds:
    #     # print(ds._mat_file_list[count])
    #     count += 1
    #     print(mat.shape, rgb.shape)
    # icvl_ft = ['Labtest_0910-1513', 'bulb_0822-0903', 'Labtest_0910-1509', 'Lehavim_0910-1717', 'Labtest_0910-1511', 'IDS_COLORCHECK_1020-1223', '4cam_0411-1648', 'objects_0924-1650', 'pepper_0503-1229', 'peppers_0503-1308', 'pepper_0503-1236', 'lehavim_0910-1610', 'plt_0411-1207', 'peppers_0503-1315', 'Lehavim_0910-1718', 'Lehavim_0910-1630']
    # icvl_train = _get_train_for_ft(icvl_ft)
    # print(icvl_train)
    # print(len(icvl_train))
    # print(len(icvl_train) / len(_get_all_train_set()))
    # test_list = _get_ft_test_set()
    # print(test_list[1], test_list[26])
    
    for i, (train, test) in enumerate(ICVL_FT_DATA.items()):
        print('%d & %s & %s \\\\'%(i+1, test, train))
