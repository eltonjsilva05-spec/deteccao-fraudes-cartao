
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def load_metrics():
    path = RESULTS_DIR / "metricas.csv"

    if not path.exists():
        raise FileNotFoundError("Arquivo results/metricas.csv não encontrado.")

    return pd.read_csv(path)


def load_threshold():
    path = RESULTS_DIR / "threshold_analysis.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Arquivo results/threshold_analysis.csv não encontrado."
        )

    return pd.read_csv(path)


def load_result_image(filename):
    path = RESULTS_DIR / filename

    if path.exists():
        return str(path)

    return None


def get_best_model(metrics):
    return metrics.loc[metrics["F1"].idxmax()]
