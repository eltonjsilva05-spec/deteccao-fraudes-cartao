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
    gerar_matriz_confusao,
    gerar_importancia_variaveis
)

import pandas as pd


def main():

    print("=" * 60)
    print("DETECÇÃO DE FRAUDES EM CARTÃO DE CRÉDITO")
    print("=" * 60)

    print("\nCarregando dataset...")

    df = carregar_dados()

    print(f"Dataset carregado com {df.shape[0]} linhas")
    print(f"Dataset possui {df.shape[1]} colunas")

    print("\nRealizando pré-processamento...")

    df = preprocessar(df)

    print("\nSeparando treino e teste...")

    X_train, X_test, y_train, y_test = dividir_dados(df)

    print("\nTreinando Logistic Regression...")

    logistic = treinar_logistic(
        X_train,
        y_train
    )

    print("\nTreinando Random Forest...")

    random_forest = treinar_random_forest(
        X_train,
        y_train
    )

    print("\nTreinando XGBoost...")

    xgboost = treinar_xgboost(
        X_train,
        y_train
    )

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

    print("\nAVALIAÇÃO DOS MODELOS\n")

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

    print("\nCOMPARAÇÃO DOS MODELOS")
    print(resultados)

    resultados.to_csv(
        "../results/metricas.csv",
        index=False
    )

    print("\nArquivo salvo: results/metricas.csv")

    melhor_modelo_info = resultados.sort_values(
        by="F1",
        ascending=False
    ).iloc[0]

    nome_melhor_modelo = melhor_modelo_info["Modelo"]

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
        f"\nMelhor modelo: {nome_melhor_modelo}"
    )

    print("\nGerando gráficos...")

    gerar_curva_roc(
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
        X_train
    )

    print(
        "Gráficos salvos na pasta results/"
    )

    print("\nProjeto finalizado com sucesso!")


if __name__ == "__main__":
    main()