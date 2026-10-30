"""EDA: rozkład klas, długości nagrań, surowy sygnał, PSD, theta/beta."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.signal import welch

from src.config import BANDS, CHANNELS, SAMPLING_RATE_HZ, WELCH_NPERSEG
from src.data import load_raw
from src.features import band_power

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

CLASS_COLORS = {"ADHD": "tab:red", "Control": "tab:blue"}


def class_distribution(df: pd.DataFrame) -> tuple[Figure, pd.Series]:
    """Liczy dzieci per klasa i rysuje rozkład klas."""
    counts = df.groupby("ID")["Class"].first().value_counts()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(counts.index, counts.to_numpy(),
           color=[CLASS_COLORS[c] for c in counts.index], edgecolor="black")
    ax.set_title("Liczba dzieci per klasa")
    ax.set_ylabel("Liczba dzieci")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    return fig, counts


def recording_length_distribution(df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]:
    """Liczy długość nagrania per dziecko w sekundach i rysuje ją per klasa."""
    lengths = df.groupby("ID").agg(n_samples=("Class", "size"), Class=("Class", "first"))
    lengths["duration_sec"] = lengths["n_samples"] / SAMPLING_RATE_HZ

    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in CLASS_COLORS.items():
        subset = lengths.loc[lengths["Class"] == cls, "duration_sec"]
        ax.hist(subset, bins=15, alpha=0.6, label=cls, color=color, edgecolor="black")
    ax.set_xlabel("Długość nagrania [s]")
    ax.set_ylabel("Liczba dzieci")
    ax.set_title("Rozkład długości nagrań per klasa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "recording_length_distribution.png", dpi=150)
    return fig, lengths


def plot_raw_signal_examples(df: pd.DataFrame, example_channels=("Fp1", "Cz", "O1"), seconds: int = 4) -> Figure:
    """Rysuje kilka sekund surowego sygnału dla jednego dziecka ADHD i jednego Control."""
    n_samples = seconds * SAMPLING_RATE_HZ
    fig, axes = plt.subplots(len(example_channels), 2, figsize=(10, 6), sharex=True)

    for col, cls in enumerate(["ADHD", "Control"]):
        child_id = df.loc[df["Class"] == cls, "ID"].iloc[0]
        child_signal = df.loc[df["ID"] == child_id].iloc[:n_samples]
        t = np.arange(len(child_signal)) / SAMPLING_RATE_HZ

        for row, ch in enumerate(example_channels):
            ax = axes[row, col]
            ax.plot(t, child_signal[ch].to_numpy(), linewidth=0.8, color=CLASS_COLORS[cls])
            ax.set_ylabel(ch)
            if row == 0:
                ax.set_title(f"{cls} (dziecko {child_id})")
            if row == len(example_channels) - 1:
                ax.set_xlabel("Czas [s]")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "raw_signal_examples.png", dpi=150)
    return fig


def compute_psd_per_child(df: pd.DataFrame) -> list[dict]:
    """Liczy PSD metodą Welcha per kanał, na całym nagraniu każdego dziecka."""
    records = []
    for child_id, group in df.groupby("ID"):
        psd_per_channel = []
        for ch in CHANNELS:
            freqs, psd = welch(group[ch].to_numpy(), fs=SAMPLING_RATE_HZ, nperseg=WELCH_NPERSEG)
            psd_per_channel.append(psd)
        records.append({
            "ID": child_id,
            "Class": group["Class"].iat[0],
            "freqs": freqs,
            "psd": np.array(psd_per_channel),  # kształt (n_channels, n_freqs)
        })
    return records


def average_psd_by_class(records: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Uśrednia PSD per kanał po dzieciach, osobno dla każdej klasy."""
    freqs = records[0]["freqs"]
    psd_by_class = {}
    for cls in ["ADHD", "Control"]:
        stacked = np.stack([r["psd"] for r in records if r["Class"] == cls])
        psd_by_class[cls] = stacked.mean(axis=0)  # kształt (n_channels, n_freqs)
    return freqs, psd_by_class


def plot_average_psd(freqs: np.ndarray, psd_by_class: dict[str, np.ndarray]) -> Figure:
    """Rysuje uśrednione po kanałach PSD dla ADHD vs Control."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in CLASS_COLORS.items():
        mean_across_channels = psd_by_class[cls].mean(axis=0)
        ax.semilogy(freqs, mean_across_channels, label=cls, color=color)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("Moc (skala logarytmiczna)")
    ax.set_title("Uśrednione widmo mocy (PSD), uśrednione po kanałach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "average_psd.png", dpi=150)
    return fig


def theta_beta_ratio_per_channel(freqs: np.ndarray, psd_by_class: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Stosunek mocy theta/beta per kanał, uśredniony po dzieciach w ramach klasy."""
    ratios = {}
    for cls, psd in psd_by_class.items():
        theta = band_power(freqs, psd, *BANDS["theta"])
        beta = band_power(freqs, psd, *BANDS["beta"])
        ratios[cls] = theta / beta
    return ratios


def plot_theta_beta_ratio(ratios: dict[str, np.ndarray]) -> Figure:
    """Rysuje stosunek theta/beta per kanał dla ADHD vs Control."""
    x = np.arange(len(CHANNELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, ratios["ADHD"], width, label="ADHD", color=CLASS_COLORS["ADHD"], edgecolor="black")
    ax.bar(x + width / 2, ratios["Control"], width, label="Control", color=CLASS_COLORS["Control"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.set_ylabel("Stosunek theta/beta")
    ax.set_title("Stosunek theta/beta per kanał per klasa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "theta_beta_ratio.png", dpi=150)
    return fig


def theta_beta_ratio_per_child(records: list[dict], channels=("Fz", "Cz", "T7")) -> pd.DataFrame:
    """Stosunek theta/beta per dziecko per kanał - do sprawdzenia, czy różnica
    między klasami pochodzi od całej grupy, czy tylko kilkorga dzieci."""
    channel_idx = {ch: i for i, ch in enumerate(CHANNELS)}
    rows = []
    for r in records:
        for ch in channels:
            psd = r["psd"][channel_idx[ch]]
            theta = band_power(r["freqs"], psd, *BANDS["theta"])
            beta = band_power(r["freqs"], psd, *BANDS["beta"])
            rows.append({"ID": r["ID"], "Class": r["Class"], "Channel": ch, "ratio": theta / beta})
    return pd.DataFrame(rows)


def plot_theta_beta_boxplot_per_child(ratio_df: pd.DataFrame, channels=("Fz", "Cz", "T7")) -> Figure:
    """Rysuje rozkłady theta/beta per dziecko (ADHD vs Control) dla podanych kanałów."""
    fig, axes = plt.subplots(1, len(channels), figsize=(10, 4), sharey=True)
    for ax, ch in zip(axes, channels):
        data = [
            ratio_df.loc[(ratio_df["Channel"] == ch) & (ratio_df["Class"] == cls), "ratio"]
            for cls in ["ADHD", "Control"]
        ]
        ax.boxplot(data, tick_labels=["ADHD", "Control"])
        ax.set_title(ch)
    axes[0].set_ylabel("Stosunek theta/beta (per dziecko)")
    fig.suptitle("Rozkład theta/beta per dziecko - Fz, Cz, T7")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "theta_beta_boxplot_per_child.png", dpi=150)
    return fig


def main() -> None:
    """Uruchamia pełną EDA: liczy statystyki, drukuje podsumowania, zapisuje wykresy."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    _, counts = class_distribution(df)
    print("Klasy (poziom dziecka):")
    print(counts.to_string())

    _, lengths = recording_length_distribution(df)
    print("\nDługość nagrania [s] - statystyki per klasa:")
    print(lengths.groupby("Class")["duration_sec"].describe().to_string())

    plot_raw_signal_examples(df)

    records = compute_psd_per_child(df)
    freqs, psd_by_class = average_psd_by_class(records)
    plot_average_psd(freqs, psd_by_class)

    ratios = theta_beta_ratio_per_channel(freqs, psd_by_class)
    plot_theta_beta_ratio(ratios)
    print("\nStosunek theta/beta per kanał per klasa (uśredniony po dzieciach):")
    ratio_table = pd.DataFrame(ratios, index=CHANNELS)
    print(ratio_table.round(3).to_string())

    per_child_ratio = theta_beta_ratio_per_child(records)
    plot_theta_beta_boxplot_per_child(per_child_ratio)
    print("\nTheta/beta per dziecko (Fz, Cz, T7) - mediana ADHD vs Control:")
    print(per_child_ratio.groupby(["Channel", "Class"])["ratio"].median().unstack().to_string())

    print(f"\nWykresy zapisane w {FIGURES_DIR}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
