import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from config import RESULTS_PATH


def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o desempenho do modelo no conjunto de teste.

    Métricas utilizadas:
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Average Precision (AP)

    O conjunto de teste mantém sua distribuição original,
    permitindo uma avaliação mais realista do problema
    de fraude.
    """

    y_pred = modelo.predict(X_test)

    print("\n" + "=" * 60)
    print(f"MODELO: {type(modelo).__name__}")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legítima",
                "Fraude"
            ],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Métricas baseadas em probabilidade
    # ---------------------------------------------------------

    roc_auc = None
    average_precision = None

    if hasattr(modelo, "predict_proba"):

        y_probs = modelo.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_probs
        )

        average_precision = average_precision_score(
            y_test,
            y_probs
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"Average Precision: {average_precision:.4f}"
        )

    # ---------------------------------------------------------
    # Métricas principais
    # ---------------------------------------------------------

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

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-Score:  {f1:.4f}"
    )

    # ---------------------------------------------------------
    # Matriz de confusão
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nMatriz de Confusão:")

    print(
        f"Verdadeiros Negativos: {tn:,}"
    )

    print(
        f"Falsos Positivos:      {fp:,}"
    )

    print(
        f"Falsos Negativos:      {fn:,}"
    )

    print(
        f"Verdadeiros Positivos: {tp:,}"
    )

    return {
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Average_Precision": average_precision
    }


def gerar_curva_roc(
    modelo,
    X_test,
    y_test
):
    """
    Gera e salva a curva ROC.
    """

    if not hasattr(
        modelo,
        "predict_proba"
    ):
        return

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    y_probs = modelo.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probs
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probs
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Aleatório"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"Curva ROC - {type(modelo).__name__}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            f"{type(modelo).__name__}_roc.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def gerar_curva_precision_recall(
    modelo,
    X_test,
    y_test
):
    """
    Gera e salva a curva Precision-Recall.

    A curva é especialmente importante em problemas
    de classificação altamente desbalanceados.
    """

    if not hasattr(
        modelo,
        "predict_proba"
    ):
        return

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    y_probs = modelo.predict_proba(
        X_test
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_probs
    )

    average_precision = average_precision_score(
        y_test,
        y_probs
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"Average Precision = "
            f"{average_precision:.4f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        f"Curva Precision-Recall - "
        f"{type(modelo).__name__}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            f"{type(modelo).__name__}_precision_recall.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def gerar_matriz_confusao(
    modelo,
    X_test,
    y_test
):
    """
    Gera e salva a matriz de confusão.
    """

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    y_pred = modelo.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legítima",
            "Fraude"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"Matriz de Confusão - "
        f"{type(modelo).__name__}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            f"{type(modelo).__name__}_confusion_matrix.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def gerar_importancia_variaveis(
    modelo,
    X_train
):
    """
    Gera e salva o gráfico de importância das variáveis
    para modelos que possuem feature_importances_.
    """

    if not hasattr(
        modelo,
        "feature_importances_"
    ):
        return

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    importancia = modelo.feature_importances_

    indices = importancia.argsort()[::-1]

    nomes = X_train.columns[indices]
    valores = importancia[indices]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        nomes,
        valores
    )

    plt.xticks(
        rotation=90
    )

    plt.xlabel(
        "Variáveis"
    )

    plt.ylabel(
        "Importância"
    )

    plt.title(
        f"Importância das Variáveis - "
        f"{type(modelo).__name__}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            "feature_importance.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()