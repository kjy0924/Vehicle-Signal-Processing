import numpy as np
from scipy.signal import welch


def compute_welch_psd(residual_window: np.ndarray, fs: float = 1.0, nperseg: int = 256):

    # 1. Welch 변환
    freqs, psd = welch(residual_window, fs=fs, nperseg=nperseg)

    # 2. Log Scaling
    log_psd = 10 * np.log10(psd + 1e-12)

    return freqs, psd, log_psd


def scale_psd_for_autoencoder(log_psd: np.ndarray, train_min: np.ndarray, train_max: np.ndarray):

    scaled_psd = (log_psd - train_min) / (train_max - train_min + 1e-12)

    return scaled_psd


def extract_batch_psd_features(residual_series: np.ndarray, window_size: int, stride: int, fs: float = 1.0, nperseg: int = 256):

    log_psd_list = []

    for i in range(0, len(residual_series) - window_size + 1, stride):
        window_data = residual_series[i: i + window_size]
        _, _, log_psd = compute_welch_psd(window_data, fs=fs, nperseg=nperseg)
        log_psd_list.append(log_psd)

    return np.array(log_psd_list)