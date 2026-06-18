import shap


def explicar_modelo(
    modelo,
    X_test
):

    explainer = shap.Explainer(
        modelo
    )

    shap_values = explainer(
        X_test.iloc[[100]]
    )

    shap.plots.bar(
        shap_values
    )
