"""Uruchamia cały pipeline: EDA, trening, ewaluacja."""
import matplotlib

matplotlib.use("Agg")

from src import eda, evaluate, train


def main() -> None:
    print("=== EDA ===")
    eda.main()

    print("\n=== Trening ===")
    train.main()

    print("\n=== Ewaluacja ===")
    evaluate.main()


if __name__ == "__main__":
    main()
