"""Wczytanie surowego zbioru EEG i cięcie go na segmenty sygnału."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CHANNELS, SAMPLING_RATE_HZ

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "adhdata.csv"


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Wczytuje pełny CSV: 19 kanałów EEG plus ID dziecka i Class."""
    return pd.read_csv(path)


def _channels_array(group: pd.DataFrame) -> np.ndarray:
    """Zwraca nagranie jednego dziecka jako tablicę (n_channels, n_samples)."""
    return group[CHANNELS].to_numpy().T


def make_windows(df: pd.DataFrame, window_sec: int = 4) -> list[dict]:
    """Tnie nagranie każdego dziecka na nienakładające się okna; niepełne okno na końcu odrzuca."""
    window_samples = window_sec * SAMPLING_RATE_HZ
    windows = []
    for child_id, group in df.groupby("ID"):
        signal = _channels_array(group)
        n_full_windows = signal.shape[1] // window_samples
        cls = group["Class"].iat[0]
        for w in range(n_full_windows):
            start = w * window_samples
            windows.append({
                "ID": child_id,
                "Class": cls,
                "window_idx": w,
                "signal": signal[:, start:start + window_samples],
            })
    return windows


def make_whole_recordings(df: pd.DataFrame) -> list[dict]:
    """Traktuje całe nagranie dziecka jako jeden segment (bez okienkowania)."""
    recordings = []
    for child_id, group in df.groupby("ID"):
        recordings.append({
            "ID": child_id,
            "Class": group["Class"].iat[0],
            "window_idx": 0,
            "signal": _channels_array(group),
        })
    return recordings
