import os


# ============================================================
# CONFIGURAÇÕES DO PROJETO
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.30


# ============================================================
# DIRETÓRIO PRINCIPAL DO PROJETO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATASET
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

DATASET_PATH = os.path.join(
    DATA_PATH,
    "creditcard.csv"
)

DATASET_URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/data/creditcard.csv"
)


# ============================================================
# MODELOS E RESULTADOS
# ============================================================

MODELS_PATH = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Class"