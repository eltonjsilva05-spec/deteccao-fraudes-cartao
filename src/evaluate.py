
import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from config import RESULTS_PATH


# ============================================================
# IDENTIDADE VISUAL — FRAUD ANALYTICS
# ============================================================

BACKGROUND = "#0b1120"
PANEL = "#111827"
GRID = "#263449"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"

CYAN = "#38bdf8"
BLUE = "#3b82f6"
PURPLE = "#a78bfa"
GREEN = "#34d399"
RED = "#fb7185"
YELLOW = "#fbbf24"

GRADIENT = LinearSegmentedColormap.from_list(
    "fraud_gradient",
    [
        "#172554",
        "#2563eb",
        "#38bdf8",
        "#34d399"
    ]
)


def configurar_grafico(figsize=(10, 6)):
    """
    Configura o padrão visual dos gráficos.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor=BACKGROUND
    )

    ax.set_facecolor(PANEL)

    ax.tick_params(
        colors=MUTED,
        labelsize=10
    )

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.grid(
        True,
        color=GRID,
        linestyle="--",
        linewidth=0.7,
        alpha=0.7
    )

    ax.set_axisbelow(True)

    return fig, ax


def salvar_grafico(fig, nome_arquivo):
    """
    Salva o gráfico mantendo o padrão visual.
    """

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    caminho = os.path.join(
        RESULTS_PATH,
        nome_arquivo
    )

    fig.savefig(
        caminho,
        dpi=240,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        edgecolor=BACKGROUND
    )

    plt.close(fig)


# ============================================================
# AVALIAÇÃO DOS MODELOS
# ============================================================

def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o desempenho do modelo no conjunto de teste.

    Métricas:
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Average Precision
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

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision: {average_precision:.4f}")

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

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nMatriz de Confusão:")
    print(f"Verdadeiros Negativos: {tn:,}")
    print(f"Falsos Positivos:      {fp:,}")
    print(f"Falsos Negativos:      {fn:,}")
    print(f"Verdadeiros Positivos: {tp:,}")

    return {
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Average_Precision": average_precision
    }


# ============================================================
# CURVA ROC
# ============================================================

def gerar_curva_roc(modelo, X_test, y_test):
    """
    Gera uma curva ROC com identidade visual Fraud Analytics.
    """

    if not hasattr(modelo, "predict_proba"):
        return

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

    fig, ax = configurar_grafico(
        figsize=(10, 6)
    )

    ax.plot(
        fpr,
        tpr,
        color=CYAN,
        linewidth=3,
        label=f"ROC-AUC: {roc_auc:.2%}"
    )

    ax.fill_between(
        fpr,
        tpr,
        color=CYAN,
        alpha=0.12
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Classificador aleatório"
    )

    ax.set_title(
        "ROC CURVE",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=11,
        labelpad=10
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=11,
        labelpad=10
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    ax.legend(
        loc="lower right",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=10
    )

    salvar_grafico(
        fig,
        f"{type(modelo).__name__}_roc.png"
    )


# ============================================================
# CURVA PRECISION-RECALL
# ============================================================

def gerar_curva_precision_recall(
    modelo,
    X_test,
    y_test
):
    """
    Gera uma curva Precision-Recall com identidade visual.
    """

    if not hasattr(modelo, "predict_proba"):
        return

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

    baseline = np.mean(y_test)

    fig, ax = configurar_grafico(
        figsize=(10, 6)
    )

    ax.plot(
        recall,
        precision,
        color=PURPLE,
        linewidth=3,
        label=f"Average Precision: {average_precision:.2%}"
    )

    ax.fill_between(
        recall,
        precision,
        color=PURPLE,
        alpha=0.12
    )

    ax.axhline(
        y=baseline,
        color=YELLOW,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Baseline: {baseline:.2%}"
    )

    ax.set_title(
        "PRECISION-RECALL CURVE",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18
    )

    ax.set_xlabel(
        "Recall",
        fontsize=11,
        labelpad=10
    )

    ax.set_ylabel(
        "Precision",
        fontsize=11,
        labelpad=10
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    ax.legend(
        loc="lower left",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=10
    )

    salvar_grafico(
        fig,
        f"{type(modelo).__name__}_precision_recall.png"
    )


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def gerar_matriz_confusao(
    modelo,
    X_test,
    y_test
):
    """
    Gera uma matriz de confusão em formato de mapa de calor.
    """

    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = configurar_grafico(
        figsize=(8, 7)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=GRADIENT
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    colorbar.ax.yaxis.set_tick_params(
        color=MUTED
    )

    plt.setp(
        colorbar.ax.yaxis.get_ticklabels(),
        color=MUTED
    )

    labels = [
        "Legítima",
        "Fraude"
    ]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Classe prevista",
        fontsize=11,
        labelpad=12
    )

    ax.set_ylabel(
        "Classe real",
        fontsize=11,
        labelpad=12
    )

    ax.set_title(
        "CONFUSION MATRIX",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color="white" if cm[i, j] > threshold else TEXT
            )

    ax.set_xticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color=BACKGROUND,
        linestyle="-",
        linewidth=3
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    salvar_grafico(
        fig,
        f"{type(modelo).__name__}_confusion_matrix.png"
    )


# ============================================================
# IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

def gerar_importancia_variaveis(
    modelo,
    X_train
):
    """
    Gera um gráfico horizontal de importância das variáveis.
    """

    if not hasattr(modelo, "feature_importances_"):
        return

    importancia = modelo.feature_importances_

    indices = importancia.argsort()

    nomes = X_train.columns[indices]
    valores = importancia[indices]

    fig, ax = configurar_grafico(
        figsize=(11, 8)
    )

    bars = ax.barh(
        nomes,
        valores,
        color=plt.cm.viridis(
            np.linspace(0.35, 0.95, len(valores))
        ),
        height=0.68
    )

    ax.set_title(
        "FEATURE IMPORTANCE",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20
    )

    ax.set_xlabel(
        "Importância",
        fontsize=11,
        labelpad=10
    )

    ax.set_ylabel(
        "Variáveis",
        fontsize=11,
        labelpad=10
    )

    ax.grid(
        axis="x",
        alpha=0.5
    )

    ax.grid(
        axis="y",
        visible=False
    )

    for bar, value in zip(bars, valores):

        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.3f}",
            va="center",
            color=TEXT,
            fontsize=8
        )

    salvar_grafico(
        fig,
        "feature_importance.png"
    )
