from data_loader import carregar_dados
from preprocessing import preprocessar

from train import (
    dividir_dados,
    treinar_logistic,
    treinar_random_forest,
    treinar_xgboost,
    salvar_modelo
)

from evaluate import (
    avaliar_modelo,
    gerar_curva_roc,
    gerar_curva_precision_recall,
    gerar_matriz_confusao,
    gerar_importancia_variaveis
)

from threshold_analysis import analisar_thresholds

from explainability import explicar_modelo

from config import (
    MODELS_PATH,
    RESULTS_PATH
)

import pandas as pd
import os


def main():

    print("=" * 60)
    print("DETECÇÃO DE FRAUDES EM CARTÃO DE CRÉDITO - V3")
    print("=" * 60)

    os.makedirs(
        RESULTS_PATH,
        exist_ok=True
    )

    os.makedirs(
        MODELS_PATH,
        exist_ok=True
    )

    print("\n[1/7] Carregando dataset...")

    df = carregar_dados()

    print(
        f"Dataset carregado com {df.shape[0]:,} linhas"
    )

    print(
        f"Dataset possui {df.shape[1]} colunas"
    )

    print("\n[2/7] Realizando pré-processamento...")

    df = preprocessar(df)

    print(
        "Pré-processamento concluído."
    )

    print("\n[3/7] Separando treino e teste...")

    (
        X_train,
        X_train_smote,
        X_test,
        y_train,
        y_train_smote,
        y_test
    ) = dividir_dados(df)

    print(
        f"\nTreino original: "
        f"{X_train.shape[0]:,} registros"
    )

    print(
        f"Treino balanceado: "
        f"{X_train_smote.shape[0]:,} registros"
    )

    print(
        f"Teste: "
        f"{X_test.shape[0]:,} registros"
    )

    print("\n[4/7] Treinando modelos...")

    print("\n→ Logistic Regression")

    logistic = treinar_logistic(
        X_train,
        y_train
    )

    print(
        "✓ Logistic Regression concluído."
    )

    print("\n→ Random Forest")

    random_forest = treinar_random_forest(
        X_train_smote,
        y_train_smote
    )

    print(
        "✓ Random Forest concluído."
    )

    print("\n→ XGBoost")

    xgboost = treinar_xgboost(
        X_train_smote,
        y_train_smote
    )

    print(
        "✓ XGBoost concluído."
    )

    print("\n[5/7] Salvando modelos...")

    salvar_modelo(
        logistic,
        "logistic"
    )

    salvar_modelo(
        random_forest,
        "random_forest"
    )

    salvar_modelo(
        xgboost,
        "xgboost"
    )

    print("\n[6/7] Avaliando modelos...\n")

    resultado_logistic = avaliar_modelo(
        logistic,
        X_test,
        y_test
    )

    resultado_rf = avaliar_modelo(
        random_forest,
        X_test,
        y_test
    )

    resultado_xgb = avaliar_modelo(
        xgboost,
        X_test,
        y_test
    )

    resultados = pd.DataFrame([
        {
            "Modelo": "Logistic Regression",
            **resultado_logistic
        },
        {
            "Modelo": "Random Forest",
            **resultado_rf
        },
        {
            "Modelo": "XGBoost",
            **resultado_xgb
        }
    ])

    print("\n" + "=" * 60)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    print(
        resultados.to_string(index=False)
    )

    caminho_metricas = os.path.join(
        RESULTS_PATH,
        "metricas.csv"
    )

    resultados.to_csv(
        caminho_metricas,
        index=False
    )

    print(
        f"\n✓ Métricas salvas em: "
        f"{caminho_metricas}"
    )

    melhor_modelo_info = resultados.sort_values(
        by="F1",
        ascending=False
    ).iloc[0]

    nome_melhor_modelo = (
        melhor_modelo_info["Modelo"]
    )

    if nome_melhor_modelo == "Logistic Regression":

        melhor_modelo = logistic

    elif nome_melhor_modelo == "Random Forest":

        melhor_modelo = random_forest

    else:

        melhor_modelo = xgboost

    salvar_modelo(
        melhor_modelo,
        "best_model"
    )

    print(
        f"\n✓ Melhor modelo: "
        f"{nome_melhor_modelo}"
    )

    print(
        f"✓ F1-Score: "
        f"{melhor_modelo_info['F1']:.4f}"
    )

    print("\nGerando gráficos...")

    gerar_curva_roc(
        xgboost,
        X_test,
        y_test
    )

    gerar_curva_precision_recall(
        xgboost,
        X_test,
        y_test
    )

    gerar_matriz_confusao(
        xgboost,
        X_test,
        y_test
    )

    gerar_importancia_variaveis(
        xgboost,
        X_train_smote
    )

    print(
        "✓ Gráficos salvos na pasta results/"
    )

    print(
        "\nAnalisando diferentes thresholds..."
    )

    melhor_threshold = analisar_thresholds(
        xgboost,
        X_test,
        y_test
    )

    print(
        f"\n✓ Threshold recomendado: "
        f"{melhor_threshold['Threshold']:.2f}"
    )

    print(
        f"✓ Precision no threshold recomendado: "
        f"{melhor_threshold['Precision']:.4f}"
    )

    print(
        f"✓ Recall no threshold recomendado: "
        f"{melhor_threshold['Recall']:.4f}"
    )

    print(
        f"✓ F1-Score no threshold recomendado: "
        f"{melhor_threshold['F1']:.4f}"
    )

    print("\nGerando explicabilidade SHAP...")

    explicar_modelo(
        xgboost,
        X_test,
        indice=100,
        nome_arquivo="shap_local.png"
    )

    print(
        "\n✓ Explicabilidade SHAP processada."
    )

    print("\n" + "=" * 60)
    print("PROJETO FINALIZADO COM SUCESSO")
    print("=" * 60)


if __name__ == "__main__":
    main()