import os

import matplotlib.pyplot as plt
import shap

from config import RESULTS_PATH


def explicar_modelo(
    modelo,
    X_test,
    indice=100,
    nome_arquivo="shap_local.png"
):
    """
    Gera uma explicação SHAP para uma transação específica.

    Parameters
    ----------
    modelo : modelo treinado
        Modelo compatível com SHAP.

    X_test : pandas.DataFrame
        Dados de teste utilizados para explicação.

    indice : int
        Índice da transação que será explicada.

    nome_arquivo : str
        Nome do arquivo salvo em results/.
    """

    if X_test.empty:
        print(
            "⚠ Não foi possível gerar explicabilidade: "
            "X_test está vazio."
        )
        return

    indice = min(
        indice,
        len(X_test) - 1
    )

    amostra = X_test.iloc[[indice]]

    print("\n" + "=" * 60)
    print("EXPLICABILIDADE SHAP")
    print("=" * 60)

    print(
        f"Analisando transação de índice: {indice}"
    )

    try:

        print(
            "\n→ Criando explainer SHAP..."
        )

        explainer = shap.Explainer(
            modelo
        )

        print(
            "→ Calculando valores SHAP..."
        )

        shap_values = explainer(
            amostra
        )

        os.makedirs(
            RESULTS_PATH,
            exist_ok=True
        )

        caminho = os.path.join(
            RESULTS_PATH,
            nome_arquivo
        )

        print(
            "→ Gerando gráfico..."
        )

        plt.figure(
            figsize=(10, 6)
        )

        shap.plots.bar(
            shap_values,
            show=False
        )

        plt.title(
            "SHAP - Importância das Variáveis "
            "na Transação"
        )

        plt.tight_layout()

        plt.savefig(
            caminho,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"✓ Explicação SHAP salva em: {caminho}"
        )

    except Exception as erro:

        print(
            f"⚠ Não foi possível gerar "
            f"a explicação SHAP: {erro}"
        )