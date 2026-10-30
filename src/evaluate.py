"""Ewaluacja: wczytuje predykcje out-of-fold i ważność cech policzone w train.py
(results/predictions.csv, results/feature_importance.csv), liczy metryki na poziomie
dziecka (balanced accuracy, recall, precision, F1, ROC AUC - średnia ± std po 5
foldach) i rysuje wykresy (macierz pomyłek, ROC, ważność cech). Nic nie trenuje."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    RocCurveDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FEATURE_IMPORTANCE_PATH, FIGURES_DIR, N_FOLDS, PREDICTIONS_PATH, RESULTS_DIR, SUMMARY_PATH

METRIC_NAMES = ["balanced_accuracy", "recall", "precision", "f1", "roc_auc"]


def compute_binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Liczy balanced accuracy, recall, precision, F1 i ROC AUC dla jednej grupy predykcji."""
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def metrics_per_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Liczy metryki osobno dla każdej kombinacji (fold, wariant, model)."""
    rows = []
    for (fold, variant, model), group in predictions.groupby(["fold", "variant", "model"]):
        metrics = compute_binary_metrics(group["y_true"], group["y_pred"], group["proba"])
        rows.append({"fold": fold, "variant": variant, "model": model, "n": len(group), **metrics})
    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Dla każdej (wariant, model): średnia ± std metryk po 5 foldach."""
    summary = per_fold.groupby(["variant", "model"])[METRIC_NAMES].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def pick_best(summary: pd.DataFrame) -> tuple[str, str]:
    """Wybiera (wariant, model) z najwyższym średnim balanced accuracy."""
    best_row = summary.loc[summary["balanced_accuracy_mean"].idxmax()]
    return best_row["variant"], best_row["model"]


def plot_confusion_matrix_for(predictions: pd.DataFrame, variant: str, model: str) -> Figure:
    """Confusion matrix dla wybranej (wariant, model): predykcje out-of-fold z 5
    foldów, każde dziecko liczone raz. W komórkach: etykieta TN/FP/FN/TP, liczba
    i procent w wierszu (czyli wśród dzieci tej samej prawdziwej klasy)."""
    subset = predictions[(predictions["variant"] == variant) & (predictions["model"] == model)]
    cm = confusion_matrix(subset["y_true"], subset["y_pred"], labels=[0, 1])
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.imshow(cm, cmap="viridis")
    ax.set_xticks([0, 1], labels=["Control", "ADHD"])
    ax.set_yticks([0, 1], labels=["Control", "ADHD"])
    ax.set_xlabel("Przewidziana klasa")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title(
        f"Confusion matrix (poziom dziecka) - {variant} / {model}\n"
        f"predykcje out-of-fold, {N_FOLDS} foldów",
        fontsize=10,
    )

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] < threshold else "black"
            ax.text(j, i, f"{cell_labels[i, j]}\n{cm[i, j]}\n({row_pct[i, j]:.1f}%)",
                     ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    return fig


def plot_roc_curve_for(predictions: pd.DataFrame, variant: str, model: str) -> Figure:
    """Krzywa ROC dla wybranej (wariant, model), na predykcjach out-of-fold z 5 foldów."""
    subset = predictions[(predictions["variant"] == variant) & (predictions["model"] == model)]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(subset["y_true"], subset["proba"], ax=ax, name=f"{variant}/{model}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Krzywa ROC (poziom dziecka)\nout-of-fold, 5 foldów")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    return fig


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15) -> Figure:
    """Rysuje top_n cech wg ważności. Przyjmuje tabelę (feature, importance) -
    działa tak samo na świeżo policzonych danych, jak i wczytanych z CSV."""
    top = importance.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"].to_numpy(), color="tab:green", edgecolor="black")
    ax.set_xlabel("Ważność cechy (im wyżej, tym więcej wnosi do podziałów w drzewach)")
    ax.set_title(f"Top {top_n} cech - RandomForest")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rf_feature_importance.png", dpi=150)
    return fig


def main() -> None:
    """Wczytuje CSV z train.py, liczy metryki, rysuje wykresy, zapisuje summary.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    summary = summarize(metrics_per_fold(predictions))
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Metryki na poziomie dziecka (średnia ± std po 5 foldach):")
    print(summary.round(3))

    best_variant, best_model = pick_best(summary)
    print(f"\nNajlepsza kombinacja (po balanced accuracy): {best_variant} / {best_model}")

    plot_confusion_matrix_for(predictions, best_variant, best_model)
    plot_roc_curve_for(predictions, best_variant, best_model)
    plot_feature_importance(importance)

    print(f"\nPodsumowanie zapisane do {SUMMARY_PATH}")
    print(f"Wykresy zapisane w {FIGURES_DIR}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
