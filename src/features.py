"""Cechy mocy pasmowej wyliczone z segmentów sygnału EEG metodą Welcha (PSD)."""
import numpy as np
import pandas as pd
from scipy.signal import welch

from src.config import BANDS, CHANNELS, SAMPLING_RATE_HZ, WELCH_NPERSEG
from src.data import load_raw, make_whole_recordings, make_windows


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    """Sumuje moc z częstotliwości należących do pasma."""
    mask = (freqs >= low) & (freqs < high)
    return psd[..., mask].sum(axis=-1)


def extract_segment_features(signal: np.ndarray) -> dict[str, float]:
    """Log mocy pasmowej (delta/theta/alpha/beta) i stosunek theta/beta, per kanał."""
    features = {}
    for ch_idx, ch in enumerate(CHANNELS):
        freqs, psd = welch(signal[ch_idx], fs=SAMPLING_RATE_HZ, nperseg=WELCH_NPERSEG)
        band_powers = {band: band_power(freqs, psd, low, high) for band, (low, high) in BANDS.items()}
        for band, power in band_powers.items():
            features[f"{ch}_{band}_logpower"] = np.log(power)
        features[f"{ch}_theta_beta_ratio"] = band_powers["theta"] / band_powers["beta"]
    return features


def build_feature_table(segments: list[dict]) -> pd.DataFrame:
    """Zamienia listę segmentów sygnału (z data.make_windows/make_whole_recordings)
    na tabelę cech z ID, Class, window_idx plus cechami per kanał."""
    rows = []
    for seg in segments:
        row = {"ID": seg["ID"], "Class": seg["Class"], "window_idx": seg["window_idx"]}
        row.update(extract_segment_features(seg["signal"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Buduje obie wersje tabeli cech i drukuje podsumowanie diagnostyczne."""
    df = load_raw()

    windows = make_windows(df)
    windows_table = build_feature_table(windows)
    print(f"Wariant okien: {len(windows)} okien, kształt tabeli cech {windows_table.shape}")

    windows_per_child = windows_table.groupby(["ID", "Class"]).size().rename("n_windows").reset_index()
    print("\nOkna per dziecko - statystyki per klasa:")
    print(windows_per_child.groupby("Class")["n_windows"].describe().to_string())

    recordings = make_whole_recordings(df)
    whole_table = build_feature_table(recordings)
    print(f"\nWariant całego nagrania: kształt tabeli cech {whole_table.shape}")

    n_feature_cols = windows_table.shape[1] - 3  # bez ID, Class, window_idx
    print(f"\nLiczba kolumn z cechami w tabeli: {n_feature_cols}")
    print("\nPierwsze wiersze (wariant okien, pierwszych 5 kolumn z cechami):")
    print(windows_table.iloc[:3, :8].to_string())


if __name__ == "__main__":
    main()
