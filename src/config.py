"""Wspólne stałe dla zbioru EEG, ekstrakcji cech i ścieżek wyników."""
from pathlib import Path

# Nie ma tego w CSV - wzięte z dokumentacji zbioru danych.
SAMPLING_RATE_HZ = 128

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}
WELCH_NPERSEG = 256

RANDOM_STATE = 42
N_FOLDS = 5

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
