import abc
import numpy as np
from torch.utils.data import Dataset

class HSSRDS(Dataset, abc.ABC):
    bands = None

class HSSRTransform(abc.ABC):
    @abc.abstractmethod
    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        pass
    