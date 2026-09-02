import os
import pandas as pd

from config import DATASET_URL, DATASET_PATH


def carregar_dados():
    """
    Carrega o dataset de transações.

    Caso o arquivo não exista localmente, realiza o download
    e salva o dataset em data/raw.
    """

    # Garante que a pasta do dataset exista
    pasta_dataset = os.path.dirname(DATASET_PATH)

    if pasta_dataset:
        os.makedirs(pasta_dataset, exist_ok=True)

    # Download caso o dataset ainda não exista
    if not os.path.exists(DATASET_PATH):

        print("Baixando dataset...")

        df = pd.read_csv(DATASET_URL)

        df.to_csv(
            DATASET_PATH,
            index=False
        )

        print("Download concluído!")

        return df

    # Carrega dataset existente
    print("Dataset encontrado localmente.")

    return pd.read_csv(DATASET_PATH)