import numpy as np


def preprocessar(df):
    """
    Realiza o pré-processamento básico dos dados.

    A transformação logarítmica é aplicada ao campo Amount.
    A normalização será realizada posteriormente, após a
    divisão entre treino e teste, para evitar data leakage.
    """

    df = df.copy()

    # Transformação logarítmica do valor da transação
    df["Amount_log"] = np.log1p(df["Amount"])

    return df