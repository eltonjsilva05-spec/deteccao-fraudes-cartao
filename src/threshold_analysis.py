import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from config import RESULTS_PATH


def analisar_thresholds(
    modelo,
    X_test,
    y_test
):
    """
    Analisa diferentes thresholds de decisão
    para o modelo de detecção de fraude.

    O objetivo é encontrar o threshold que oferece
    o melhor equilíbrio entre Precision e Recall,
    utilizando o F1-Score como critério principal.

    Os thresholds são avaliados de 0.01 até 0.99.
    """

    print("\n" + "=" * 60)
    print("ANÁLISE DE THRESHOLDS - XGBOOST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Probabilidades previstas
    # ---------------------------------------------------------

    y_probs = modelo.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # Thresholds
    # ---------------------------------------------------------

    thresholds = [
        round(valor / 100, 2)
        for valor in range(1, 100)
    ]

    resultados = []

    # ---------------------------------------------------------
    # Avaliação de cada threshold
    # ---------------------------------------------------------

    for threshold in thresholds:

        y_pred = (
            y_probs >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        resultados.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    # ---------------------------------------------------------
    # Melhor threshold
    # ---------------------------------------------------------

    melhor_resultado = max(
        resultados,
        key=lambda resultado: resultado["F1"]
    )

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    # ---------------------------------------------------------
    # Exibição dos principais resultados
    # ---------------------------------------------------------

    print("\nMelhores thresholds por F1-Score:")

    melhores = (
        df_resultados
        .sort_values(
            by="F1",
            ascending=False
        )
        .head(10)
    )

    print(
        melhores.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format
            }
        )
    )

    # ---------------------------------------------------------
    # Resultado recomendado
    # ---------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"Melhor Threshold: "
        f"{melhor_resultado['Threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{melhor_resultado['Precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{melhor_resultado['Recall']:.4f}"
    )

    print(
        f"F1-Score: "
        f"{melhor_resultado['F1']:.4f}"
    )

    # ---------------------------------------------------------
    # Preparar diretório de resultados
    # ---------------------------------------------------------

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Salvar análise completa
    # ---------------------------------------------------------

    caminho_csv = os.path.join(
        RESULTS_PATH,
        "threshold_analysis.csv"
    )

    df_resultados.to_csv(
        caminho_csv,
        index=False
    )

    print(
        f"\n✓ Análise completa salva em: "
        f"{caminho_csv}"
    )

    # ---------------------------------------------------------
    # Gerar gráfico
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df_resultados["Threshold"],
        df_resultados["Precision"],
        label="Precision"
    )

    plt.plot(
        df_resultados["Threshold"],
        df_resultados["Recall"],
        label="Recall"
    )

    plt.plot(
        df_resultados["Threshold"],
        df_resultados["F1"],
        label="F1-Score"
    )

    # Destacar melhor threshold

    plt.scatter(
        melhor_resultado["Threshold"],
        melhor_resultado["F1"],
        s=80,
        label=(
            f"Melhor threshold = "
            f"{melhor_resultado['Threshold']:.2f}"
        )
    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Análise de Threshold - XGBoost"
    )

    plt.xticks(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ]
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    caminho_grafico = os.path.join(
        RESULTS_PATH,
        "threshold_analysis.png"
    )

    plt.savefig(
        caminho_grafico,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Gráfico salvo em: "
        f"{caminho_grafico}"
    )

    # ---------------------------------------------------------
    # Retorno
    # ---------------------------------------------------------

    return melhor_resultado