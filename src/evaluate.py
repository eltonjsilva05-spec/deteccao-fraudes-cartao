import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)


def avaliar_modelo(modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)

    print("\n" + "=" * 50)
    print(type(modelo).__name__)
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    roc_auc = None

    if hasattr(modelo, "predict_proba"):

        y_probs = modelo.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_probs
        )

        print("ROC AUC:", roc_auc)

    return {
        "Recall": recall_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred
        ),
        "F1": f1_score(
            y_test,
            y_pred
        ),
        "ROC_AUC": roc_auc
    }


def gerar_curva_roc(modelo, X_test, y_test):

    if not hasattr(modelo, "predict_proba"):
        return

    os.makedirs("../results", exist_ok=True)

    y_probs = modelo.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probs
    )

    plt.figure(figsize=(8, 5))

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve - {type(modelo).__name__}"
    )

    plt.savefig(
        f"../results/{type(modelo).__name__}_roc.png"
    )

    plt.close()


def gerar_matriz_confusao(
    modelo,
    X_test,
    y_test
):

    os.makedirs("../results", exist_ok=True)

    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        f"Matriz de Confusão - {type(modelo).__name__}"
    )

    plt.savefig(
        f"../results/{type(modelo).__name__}_confusion_matrix.png"
    )

    plt.close()


def gerar_importancia_variaveis(
    modelo,
    X_train
):

    if not hasattr(
        modelo,
        "feature_importances_"
    ):
        return

    os.makedirs(
        "../results",
        exist_ok=True
    )

    importancia = modelo.feature_importances_

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        X_train.columns,
        importancia
    )

    plt.xticks(
        rotation=90
    )

    plt.title(
        "Importância das Variáveis"
    )

    plt.tight_layout()

    plt.savefig(
        "../results/feature_importance.png"
    )

    plt.close()