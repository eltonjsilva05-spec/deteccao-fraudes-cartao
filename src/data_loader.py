import os
import pandas as pd

from config import DATASET_URL, DATASET_PATH


def carregar_dados():
    """
    Faz download do dataset caso ele não exista.
    """

    if not os.path.exists(DATASET_PATH):

        print("Baixando dataset...")

        df = pd.read_csv(DATASET_URL)

        df.to_csv(DATASET_PATH, index=False)

        print("Download concluído!")

    return pd.read_csv(DATASET_PATH)
