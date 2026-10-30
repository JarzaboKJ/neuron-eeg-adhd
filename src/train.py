"""Trening z walidacją krzyżową: LogisticRegression i RandomForest, na obu wariantach
cech (okna i całe nagranie), ze wspólnym podziałem dzieci na foldy. Zapisuje predykcje
out-of-fold do results/predictions.csv oraz ważność cech jednego RandomForest
wytrenowanego na wszystkich dzieciach do results/feature_importance.csv. evaluate.py
tylko wczytuje te pliki - samo nic nie trenuje."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_IMPORTANCE_PATH, N_FOLDS, PREDICTIONS_PATH, RANDOM_STATE, RESULTS_DIR
from src.data import load_raw, make_whole_recordings, make_windows
from src.features import build_feature_table

NON_FEATURE_COLS = {"ID", "Class", "window_idx", "fold"}
MODEL_NAMES = ["logreg", "random_forest"]


def build_pipeline(model_name: str) -> Pipeline:
    """Buduje pipeline modelu; nazwa kroku 'clf' mówi, do którego kroku trafia sample_weight."""
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
    if model_name == "random_forest":
        return Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    raise ValueError(f"Nieznany model: {model_name}")


def assign_child_folds(child_table: pd.DataFrame) -> dict[str, int]:
    """Najpierw dzielę dzieci na foldy, potem każde okno dostaje fold swojego dziecka."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    id_to_fold = {}
    for fold, (_, test_idx) in enumerate(cv.split(child_table, child_table["Class"])):
        for idx in test_idx:
            child_id = child_table["ID"].iloc[idx]
            id_to_fold[child_id] = fold
    return id_to_fold


def _feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in NON_FEATURE_COLS]


def child_weights(table: pd.DataFrame) -> pd.Series:
    """Każde okno dostaje wagę 1 / liczba okien jego dziecka."""
    return 1 / table.groupby("ID")["ID"].transform("count")


def verify_sample_weight_reaches_model(table: pd.DataFrame) -> None:
    """Trenuje ten sam pipeline raz z wagami jednostkowymi, raz z realnymi wagami per dziecko,
    dla obu modeli, i sprawdza, że predykcje się różnią - dowód, że sample_weight dociera
    do estymatora przez pipeline, a nie jest po cichu przyjęty i zgubiony po drodze."""
    feature_cols = _feature_columns(table)
    X = table[feature_cols]
    y = (table["Class"] == "ADHD").astype(int)
    real_weight = child_weights(table)

    for model_name in ["logreg", "random_forest"]:
        pipeline_uniform = build_pipeline(model_name)
        pipeline_uniform.fit(X, y, clf__sample_weight=np.ones(len(y)))

        pipeline_weighted = build_pipeline(model_name)
        pipeline_weighted.fit(X, y, clf__sample_weight=real_weight)

        proba_uniform = pipeline_uniform.predict_proba(X)[:, 1]
        proba_weighted = pipeline_weighted.predict_proba(X)[:, 1]
        assert not np.allclose(proba_uniform, proba_weighted), (
            f"sample_weight nie zmienia predykcji dla modelu {model_name} - nie dociera do estymatora"
        )
        max_diff = np.abs(proba_uniform - proba_weighted).max()
        print(f"  {model_name:14s} predykcje się różnią (max |delta| = {max_diff:.4f}) -> wagi docierają do modelu")


def run_cv(
    table: pd.DataFrame,
    model_name: str,
    id_to_fold: dict[str, int],
    use_sample_weight: bool,
) -> pd.DataFrame:
    """Uruchamia N_FOLDS-krotną walidację krzyżową i zwraca przewidywania out-of-fold
    na poziomie dziecka."""
    table = table.copy()
    table["fold"] = table["ID"].map(id_to_fold)
    feature_cols = _feature_columns(table)
    y = (table["Class"] == "ADHD").astype(int)
    weights = child_weights(table) if use_sample_weight else pd.Series(1.0, index=table.index)

    oof_rows = []
    for fold in range(N_FOLDS):
        train_mask = table["fold"] != fold
        test_mask = table["fold"] == fold

        train_ids = set(table.loc[train_mask, "ID"])
        test_ids = set(table.loc[test_mask, "ID"])
        assert train_ids.isdisjoint(test_ids), f"Wyciek danych pomiędzy dziećmi w foldzie {fold}: {train_ids & test_ids}"

        pipeline = build_pipeline(model_name)
        pipeline.fit(
            table.loc[train_mask, feature_cols],
            y[train_mask],
            clf__sample_weight=weights[train_mask],
        )
        proba = pipeline.predict_proba(table.loc[test_mask, feature_cols])[:, 1]

        oof_rows.append(pd.DataFrame({
            "ID": table.loc[test_mask, "ID"].to_numpy(),
            "Class": table.loc[test_mask, "Class"].to_numpy(),
            "fold": fold,
            "proba": proba,
        }))

    window_level = pd.concat(oof_rows, ignore_index=True)
    window_level["y_true"] = (window_level["Class"] == "ADHD").astype(int)
    window_level["y_pred"] = (window_level["proba"] >= 0.5).astype(int)

    child_level = window_level.groupby(["ID", "Class", "fold"], as_index=False)["proba"].mean()
    child_level["y_true"] = (child_level["Class"] == "ADHD").astype(int)
    child_level["y_pred"] = (child_level["proba"] >= 0.5).astype(int)
    return child_level


def run_all_combinations(windows_table: pd.DataFrame, whole_table: pd.DataFrame) -> pd.DataFrame:
    """Uruchamia 5-fold CV dla obu modeli i obu wariantów, z jednym wspólnym
    przydziałem dzieci do foldów. Zwraca predykcje out-of-fold na poziomie dziecka."""
    child_table = whole_table[["ID", "Class"]].drop_duplicates().reset_index(drop=True)
    id_to_fold = assign_child_folds(child_table)
    variants = {"windows": (windows_table, True), "whole_recording": (whole_table, False)}

    rows = []
    for variant_name, (table, use_weight) in variants.items():
        for model_name in MODEL_NAMES:
            child_level = run_cv(table, model_name, id_to_fold, use_weight)
            child_level = child_level.copy()
            child_level["variant"] = variant_name
            child_level["model"] = model_name
            rows.append(child_level)

    return pd.concat(rows, ignore_index=True)


def compute_rf_importance(table: pd.DataFrame) -> pd.DataFrame:
    """Trenuje jeden RandomForest na wszystkich dzieciach (wariant okien, z wagami)
    i zwraca ważność cech, posortowaną malejąco."""
    feature_cols = _feature_columns(table)
    y = (table["Class"] == "ADHD").astype(int)
    weights = child_weights(table)

    pipeline = build_pipeline("random_forest")
    pipeline.fit(table[feature_cols], y, clf__sample_weight=weights)
    importance = pipeline.named_steps["clf"].feature_importances_

    return (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    """Uruchamia CV dla obu modeli na obu wariantach danych i zapisuje predykcje
    oraz ważność cech do CSV w results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    windows_table = build_feature_table(make_windows(df))
    whole_table = build_feature_table(make_whole_recordings(df))

    print("Sprawdzenie, czy sample_weight dociera do modelu:")
    verify_sample_weight_reaches_model(windows_table)

    predictions = run_all_combinations(windows_table, whole_table)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredykcje out-of-fold zapisane do {PREDICTIONS_PATH}")

    importance = compute_rf_importance(windows_table)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Ważność cech RandomForest (wariant windows) zapisana do {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
