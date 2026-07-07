import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, resample
from scipy.ndimage import label
from scipy.fft import rfft, rfftfreq
import tdt

def read_tdt(folder_path: str) -> dict:
    """Reads TDT block data using the tdt library and extracts epocs and streams."""
    tdt_block = tdt.read_block(folder_path)
    data = {}

    epoc_keys = list(tdt_block.epocs.__dict__.keys())
    stream_keys = list(tdt_block.streams.__dict__.keys())

    for key in epoc_keys:
        data[key] = tdt_block.epocs[key].__dict__

    for key in stream_keys:
        data[key] = tdt_block.streams[key].__dict__

    return data


def polyphase_resample(epoch_data, fs_old, fs_new=1000.0):
    """Resamples a 1D signal using scipy polyphase resampling."""
    num_samples_new = int(round(len(epoch_data) * fs_new / fs_old))
    resampled_data = resample(epoch_data, num_samples_new)
    return resampled_data


def apply_notch_filter(sig, fs, freq=50.0, Q=30.0):
    """Applies a zero-phase narrow-band notch filter using filtfilt."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, sig)


def calculate_eeg_features(sig: np.ndarray, fs: float, win_s: float = 1.0, step_s: float = 0.5):
    """Calculates power envelope and moving spectral features (SEF95, HF Noise)."""
    win_samples = int(win_s * fs)
    step_samples = int(step_s * fs)
    
    times = []
    features = {
        'power': [],
        'sef95': [],
        'hf_noise': []
    }

    # 1. Total Power (Time-domain envelope)
    # We use a 250ms smoothing for the power.
    env = pd.Series(sig**2).rolling(
        window=int(0.25 * fs), center=True, min_periods=1).mean().to_numpy()
    
    # 2. Spectral Features (Moving Window)
    for i in range(0, len(sig) - win_samples, step_samples):
        segment = sig[i:i+win_samples]
        if np.any(np.isnan(segment)):
            features['sef95'].append(np.nan)
            features['hf_noise'].append(np.nan)
        else:
            # FFT
            fft_vals = np.abs(rfft(segment))
            freqs = rfftfreq(win_samples, 1/fs)
            
            # SEF95 (below 40Hz)
            mask_40 = freqs <= 40
            f_40 = freqs[mask_40]
            p_40 = fft_vals[mask_40]**2
            cum_p = np.cumsum(p_40)
            if cum_p[-1] > 0:
                idx_95 = np.searchsorted(cum_p, 0.95 * cum_p[-1])
                features['sef95'].append(f_40[idx_95])
            else:
                features['sef95'].append(0)
                
            # HF Noise (>100Hz)
            mask_hf = freqs > 100
            if np.any(mask_hf):
                features['hf_noise'].append(np.mean(fft_vals[mask_hf]**2))
            else:
                features['hf_noise'].append(0)
        
        times.append((i + win_samples/2) / fs)

    # Convert to arrays
    feat_dict = {k: np.array(v) for k, v in features.items()}
    feat_dict['time'] = np.array(times)
    
    # Interpolate spectral features to match original signal length
    t_full = np.arange(len(sig)) / fs
    dict_interp = {
        'power_env': env,
        'sef95': np.interp(t_full, feat_dict['time'], feat_dict['sef95']),
        'hf_noise': np.interp(t_full, feat_dict['time'], feat_dict['hf_noise'])
    }
    
    # Spectral Jitter (Std of SEF95 in a 5s window)
    sef_series = pd.Series(dict_interp['sef95'])
    dict_interp['sef_jitter'] = sef_series.rolling(window=int(5*fs), center=True).std().fillna(0).to_numpy()
    
    return dict_interp


def apply_asymmetric_heuristics(mask, fs, min_supp_sec=0.5, min_burst_sec=0.1):
    """
    Cleans the binary mask to ensure neurophysiological plausibility.
    - Removes 'bursts' that are too short to be real brain activity (noise).
    - Bridges 'suppressions' that are too short (likely just a zero-crossing).
    """
    clean_mask = mask.copy()
    
    # Temporarily mask NaNs (Artifacts)
    nan_mask = np.isnan(mask)
    clean_mask[nan_mask] = 0 # Treat as suppression for labeling
    
    min_supp_samples = int(min_supp_sec * fs)
    min_burst_samples = int(min_burst_sec * fs)

    # 1. Remove Short BURSTS
    labeled_bursts, num_bursts = label(clean_mask)
    if num_bursts > 0:
        sizes = np.bincount(labeled_bursts.ravel())
        for i in range(1, num_bursts + 1):
            if sizes[i] < min_burst_samples:
                clean_mask[labeled_bursts == i] = 0

    # 2. Remove Short SUPPRESSIONS
    inverted_mask = 1 - clean_mask
    labeled_supp, num_supp = label(inverted_mask)
    if num_supp > 0:
        sizes = np.bincount(labeled_supp.ravel())
        for i in range(1, num_supp + 1):
            if sizes[i] < min_supp_samples:
                clean_mask[labeled_supp == i] = 1

    # Restore NaNs
    clean_mask = clean_mask.astype(float)
    clean_mask[nan_mask] = np.nan
    
    return clean_mask


def load_central_manual_thresholds() -> dict:
    """Loads saved manual thresholds from centralized JSON in code directory."""
    p = Path(__file__).parent / "manual_thresholds.json"
    if p.exists():
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read central manual thresholds: {e}")
            return {}
    return {}
