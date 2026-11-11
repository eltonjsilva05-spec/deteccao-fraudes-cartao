
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

from config import MODELS_PATH, RESULTS_PATH

import pandas as pd
import os


def main():

    print("=" * 60)
    print("DETECÇÃO DE FRAUDES EM CARTÃO DE CRÉDITO - V3")
    print("=" * 60)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("\n[1/7] Carregando dataset...")

    df = carregar_dados()

    print(f"Dataset carregado com {df.shape[0]:,} linhas")
    print(f"Dataset possui {df.shape[1]} colunas")

    print("\n[2/7] Realizando pré-processamento...")

    df = preprocessar(df)

    print("Pré-processamento concluído.")

    print("\n[3/7] Separando treino e teste...")

    (
        X_train,
        X_train_smote,
        X_test,
        y_train,
        y_train_smote,
        y_test
    ) = dividir_dados(df)

    print(f"\nTreino original: {X_train.shape[0]:,} registros")
    print(f"Treino balanceado: {X_train_smote.shape[0]:,} registros")
    print(f"Teste: {X_test.shape[0]:,} registros")

    print("\n[4/7] Treinando modelos...")

    print("\n→ Logistic Regression")

    logistic = treinar_logistic(X_train, y_train)

    print("✓ Logistic Regression concluído.")

    print("\n→ Random Forest")

    random_forest = treinar_random_forest(
        X_train_smote,
        y_train_smote
    )

    print("✓ Random Forest concluído.")

    print("\n→ XGBoost")

    xgboost = treinar_xgboost(
        X_train_smote,
        y_train_smote
    )

    print("✓ XGBoost concluído.")

    print("\n[5/7] Salvando modelos...")

    salvar_modelo(logistic, "logistic")
    salvar_modelo(random_forest, "random_forest")
    salvar_modelo(xgboost, "xgboost")

    print("\n[6/7] Avaliando modelos...\n")

    resultado_logistic = avaliar_modelo(logistic, X_test, y_test)
    resultado_rf = avaliar_modelo(random_forest, X_test, y_test)
    resultado_xgb = avaliar_modelo(xgboost, X_test, y_test)

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

    print(resultados.to_string(index=False))

    caminho_metricas = os.path.join(
        RESULTS_PATH,
        "metricas.csv"
    )

    resultados.to_csv(caminho_metricas, index=False)

    print(f"\n✓ Métricas salvas em: {caminho_metricas}")

    # Identifica o modelo com maior F1-Score
    melhor_modelo_info = resultados.loc[
        resultados["F1"].idxmax()
    ]

    nome_melhor_modelo = melhor_modelo_info["Modelo"]

    modelos = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost
    }

    melhor_modelo = modelos[nome_melhor_modelo]

    salvar_modelo(melhor_modelo, "best_model")

    print(f"\n✓ Melhor modelo: {nome_melhor_modelo}")
    print(f"✓ F1-Score: {melhor_modelo_info['F1']:.4f}")

    # ========================================================
    # GERAÇÃO DOS GRÁFICOS INDIVIDUAIS
    # ========================================================

    print("\nGerando gráficos individuais dos modelos...")

    modelos_avaliados = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost
    }

    for nome, modelo in modelos_avaliados.items():

        print(f"\n→ Gerando gráficos: {nome}")

        gerar_curva_roc(
            modelo,
            X_test,
            y_test
        )

        gerar_curva_precision_recall(
            modelo,
            X_test,
            y_test
        )

        gerar_matriz_confusao(
            modelo,
            X_test,
            y_test
        )

        print(f"✓ Gráficos de {nome} gerados.")

    # Importância das variáveis permanece no XGBoost
    # para manter a análise atual do projeto.
    gerar_importancia_variaveis(
        xgboost,
        X_train_smote
    )

    print("\n✓ Gráficos salvos na pasta results/")

    # ========================================================
    # ANÁLISE DE THRESHOLD
    # ========================================================

    print("\nAnalisando diferentes thresholds...")

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

    # ========================================================
    # EXPLICABILIDADE SHAP
    # ========================================================

    print("\nGerando explicabilidade SHAP...")

    explicar_modelo(
        xgboost,
        X_test,
        indice=100,
        nome_arquivo="shap_local.png"
    )

    print("\n✓ Explicabilidade SHAP processada.")

    print("\n" + "=" * 60)
    print("PROJETO FINALIZADO COM SUCESSO")
    print("=" * 60)


if __name__ == "__main__":
    main()
