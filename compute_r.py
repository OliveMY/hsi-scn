import cv2
import numpy as np
import pandas as pd
import os
import glob


import numpy as np

def compute_ause_mse(mse: np.ndarray, uncertainty: np.ndarray, num_bins: int = 100) -> float:
    """
    Compute AUSE (Area Under Sparsification Error) using squared errors (MSE).
    
    Args:
        mse (np.ndarray): Per-sample squared errors, shape (N,)
        uncertainty (np.ndarray): Per-sample uncertainty scores, shape (N,)
        num_bins (int): Number of sparsification levels (default: 100)

    Returns:
        float: MSE-based AUSE
    """
    mse = np.asarray(mse)
    uncertainty = np.asarray(uncertainty)
    assert mse.shape == uncertainty.shape, "Shape mismatch"

    N = len(mse)
    full_mse = np.mean(mse)

    # Sort indices
    sorted_by_mse = np.argsort(mse)              # Oracle: highest MSE first
    sorted_by_uncertainty = np.argsort(uncertainty)  # Model: highest uncertainty first

    oracle_curve = []
    model_curve = []

    for i in range(num_bins + 1):
        alpha = i / num_bins
        keep_n = int((1 - alpha) * N)

        idx_oracle = sorted_by_mse[:keep_n]
        idx_model = sorted_by_uncertainty[:keep_n]

        mse_oracle = np.mean(mse[idx_oracle]) if keep_n > 0 else 0
        mse_model = np.mean(mse[idx_model]) if keep_n > 0 else 0

        oracle_curve.append(mse_oracle / full_mse)
        model_curve.append(mse_model / full_mse)

    # Compute sparsification error curve
    sparsification_error = np.array(model_curve) - np.array(oracle_curve)

    # Integrate using trapezoidal rule
    ause_mse = np.trapz(sparsification_error, dx=1 / num_bins)
    return ause_mse


AD_result_dir = '/mnt/data/ARAD/AnomalyDetect/EfficientAD/results/raw_results/'
AD_result_dir = 'qbc_result'
arad_result_dir = '1211ARAD_RE'

ad_files = sorted(glob.glob(os.path.join(AD_result_dir, '*.npy')))
cdl_files = sorted(glob.glob(os.path.join(arad_result_dir, '*.npy')))
print(cdl_files)

assert len(ad_files)+1 == len(cdl_files)

all_ad = []
all_cdl = []

for i in range(len(ad_files)):
    ad_np = np.load(ad_files[i])
    # ad_np = cv2.resize(ad_np, (512, 482))
    all_ad.append(ad_np.reshape(-1))

    cdl_np = np.load(cdl_files[i])
    all_cdl.append(cdl_np.reshape(-1))

all_ad = np.concatenate(all_ad)
all_cdl = np.concatenate(all_cdl)

ause_mse = compute_ause_mse(all_cdl, all_ad, 100)

ad_pd = pd.Series(all_ad)
cdl_pd = pd.Series(all_cdl)


print("Pearson ", ad_pd.corr(cdl_pd, 'pearson'))
print("Spearsman's: ", ad_pd.corr(cdl_pd, 'spearman'))
print("Kendall's: ", ad_pd.corr(cdl_pd, 'kendall'))
print("AUSE_MSE: ", ause_mse)