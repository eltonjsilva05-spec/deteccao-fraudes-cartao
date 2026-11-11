
from pathlib import Path

import streamlit as st
import pandas as pd

from metrics import (
    load_metrics,
    load_threshold,
    load_result_image,
)

from charts import (
    model_comparison_chart,
    model_comparison_3d_chart,
    threshold_chart,
    average_precision_chart,
)

from styles import apply_styles


# ============================================================
# CONFIGURAÇÕES DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

apply_styles(st)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

try:
    metrics = load_metrics()
    threshold_data = load_threshold()

    if metrics is None or metrics.empty:
        st.error("Não foi possível carregar as métricas dos modelos.")
        st.stop()

    required_columns = [
        "Modelo",
        "F1",
        "Precision",
        "Recall",
        "ROC_AUC",
        "Average_Precision",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing_columns:
        st.error(
            f"Colunas ausentes no arquivo de métricas: {missing_columns}"
        )
        st.stop()

except Exception as error:
    st.error(f"Erro ao carregar os dados: {error}")
    st.stop()


# ============================================================
# IDENTIFICAÇÃO DO MELHOR MODELO
# ============================================================

best_row = metrics.loc[metrics["F1"].idxmax()]

best_model = str(best_row["Modelo"])
best_f1 = float(best_row["F1"])
best_precision = float(best_row["Precision"])
best_recall = float(best_row["Recall"])


# ============================================================
# CONFIGURAÇÃO DOS ARQUIVOS DE CADA MODELO
# ============================================================

model_images = {
    "Logistic Regression": {
        "confusion": "LogisticRegression_confusion_matrix.png",
        "roc": "LogisticRegression_roc.png",
        "precision_recall": "LogisticRegression_precision_recall.png",
    },
    "Random Forest": {
        "confusion": "RandomForestClassifier_confusion_matrix.png",
        "roc": "RandomForestClassifier_roc.png",
        "precision_recall": "RandomForestClassifier_precision_recall.png",
    },
    "XGBoost": {
        "confusion": "XGBClassifier_confusion_matrix.png",
        "roc": "XGBClassifier_roc.png",
        "precision_recall": "XGBClassifier_precision_recall.png",
    },
}


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.title("🛡️ FRAUD ANALYTICS")

    st.caption("MACHINE LEARNING PLATFORM")

    st.divider()

    st.subheader("Sobre o projeto")

    st.write(
        """
        Plataforma de análise de transações financeiras
        utilizando Machine Learning para identificação
        de possíveis fraudes em cartões de crédito.
        """
    )

    st.divider()

    st.subheader("Modelos avaliados")

    st.markdown(
        """
        - Logistic Regression
        - Random Forest
        - XGBoost
        """
    )

    st.divider()

    st.subheader("Tecnologias")

    st.markdown(
        """
        - Python
        - Pandas
        - Scikit-learn
        - XGBoost
        - SHAP
        - Streamlit
        - Plotly
        """
    )

    st.divider()

    st.caption("Projeto de portfólio | 2026")


# ============================================================
# CABEÇALHO PRINCIPAL
# ============================================================

with st.container(border=True):

    col_title, col_brand = st.columns([3, 1])

    with col_title:

        st.caption("RISK INTELLIGENCE PLATFORM")

        st.title("Fraud Detection Analytics")

        st.write(
            """
            Análise inteligente de transações financeiras,
            comparação de modelos preditivos e identificação
            de padrões associados a fraudes em cartões.
            """
        )

    with col_brand:

        with st.container(border=True):

            st.markdown("## 🛡️")

            st.markdown("**FRAUD ANALYTICS**")

            st.caption("MACHINE LEARNING")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("DATA ANALYTICS")

    with col2:
        st.caption("MACHINE LEARNING")

    with col3:
        st.caption("FRAUD DETECTION")


# ============================================================
# INDICADORES PRINCIPAIS
# ============================================================

st.subheader("Indicadores de desempenho")

st.caption(
    "Resumo dos principais resultados obtidos na avaliação dos modelos."
)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:

    st.metric(
        label="🏆 Melhor modelo",
        value=best_model,
        help="Modelo com maior F1-Score.",
        border=True,
    )

with col2:

    st.metric(
        label="🎯 F1-Score",
        value=f"{best_f1:.2%}",
        help="Equilíbrio entre Precision e Recall.",
        border=True,
    )

with col3:

    st.metric(
        label="🔍 Precision",
        value=f"{best_precision:.2%}",
        help="Precisão das fraudes identificadas.",
        border=True,
    )

with col4:

    st.metric(
        label="🛡️ Recall",
        value=f"{best_recall:.2%}",
        help="Proporção das fraudes reais identificadas.",
        border=True,
    )


st.divider()


# ============================================================
# ABAS DO DASHBOARD
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visão Geral",
        "🤖 Modelos",
        "🎯 Threshold",
        "🔎 Explicabilidade",
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with tab1:

    st.subheader("Visão geral do desempenho")

    st.write(
        """
        Comparação dos principais indicadores dos modelos
        utilizados na detecção de fraudes.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            model_comparison_chart(metrics),
            width="stretch",
            key="overview_model_comparison",
        )

    with col2:

        st.plotly_chart(
            average_precision_chart(metrics),
            width="stretch",
            key="overview_average_precision",
        )

    st.subheader("Tabela comparativa")

    st.dataframe(
        metrics,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# ABA 2 — MODELOS
# ============================================================

with tab2:

    st.subheader("Explorador de modelos")

    st.write(
        """
        Selecione um algoritmo para visualizar seus indicadores
        e resultados individuais na detecção de fraudes.
        """
    )

    available_models = [
        model
        for model in metrics["Modelo"].tolist()
        if model in model_images
    ]

    if not available_models:

        st.warning("Nenhum modelo reconhecido foi encontrado.")

    else:

        default_index = (
            available_models.index("XGBoost")
            if "XGBoost" in available_models
            else 0
        )

        selected_model = st.selectbox(
            "Selecione o modelo para análise:",
            options=available_models,
            index=default_index,
            key="selected_model",
        )

        selected_row = metrics.loc[
            metrics["Modelo"] == selected_model
        ].iloc[0]

        selected_images = model_images[selected_model]

        st.divider()

        st.subheader(f"Desempenho — {selected_model}")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "F1-Score",
                f"{selected_row['F1']:.2%}",
                border=True,
            )

        with col2:
            st.metric(
                "Precision",
                f"{selected_row['Precision']:.2%}",
                border=True,
            )

        with col3:
            st.metric(
                "Recall",
                f"{selected_row['Recall']:.2%}",
                border=True,
            )

        with col4:
            st.metric(
                "ROC-AUC",
                f"{selected_row['ROC_AUC']:.2%}",
                border=True,
            )

        with col5:
            st.metric(
                "Average Precision",
                f"{selected_row['Average_Precision']:.2%}",
                border=True,
            )

        st.divider()

        st.subheader("Matriz de confusão")

        confusion_matrix = load_result_image(
            selected_images["confusion"]
        )

        if confusion_matrix is not None:

            st.image(
                confusion_matrix,
                width="stretch",
            )

        else:

            st.warning(
                f"Matriz de confusão de {selected_model} não encontrada."
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Curva ROC")

            roc_curve = load_result_image(
                selected_images["roc"]
            )

            if roc_curve is not None:

                st.image(
                    roc_curve,
                    width="stretch",
                )

            else:

                st.warning("Curva ROC não encontrada.")

        with col2:

            st.subheader("Curva Precision-Recall")

            precision_recall = load_result_image(
                selected_images["precision_recall"]
            )

            if precision_recall is not None:

                st.image(
                    precision_recall,
                    width="stretch",
                )

            else:

                st.warning("Curva Precision-Recall não encontrada.")

        st.divider()

        st.subheader("Comparação dos três modelos")

        st.plotly_chart(
            model_comparison_3d_chart(metrics),
            width="stretch",
            key="models_comparison_3d",
        )


# ============================================================
# ABA 3 — THRESHOLD
# ============================================================

with tab3:

    st.subheader("Análise de Threshold")

    st.caption("Análise realizada com o modelo XGBoost.")

    st.write(
        """
        Avaliação do impacto do limiar de decisão
        no equilíbrio entre Precision e Recall.
        """
    )

    if threshold_data is not None and not threshold_data.empty:

        st.plotly_chart(
            threshold_chart(threshold_data),
            width="stretch",
            key="threshold_analysis",
        )

        st.subheader("Dados de Threshold")

        st.dataframe(
            threshold_data,
            width="stretch",
            hide_index=True,
        )

    else:

        st.warning("Dados de Threshold não encontrados.")

    st.divider()

    st.subheader("Curva Precision-Recall — XGBoost")

    precision_recall = load_result_image(
        "XGBClassifier_precision_recall.png"
    )

    if precision_recall is not None:

        st.image(
            precision_recall,
            width="stretch",
        )

    else:

        st.warning("Imagem Precision-Recall não encontrada.")


# ============================================================
# ABA 4 — EXPLICABILIDADE
# ============================================================

with tab4:

    st.subheader("Explicabilidade do modelo")

    st.caption("Análises de importância e SHAP realizadas com XGBoost.")

    st.write(
        """
        Análise das variáveis que mais influenciam
        as previsões do modelo de detecção de fraudes.
        """
    )

    st.divider()

    st.subheader("Importância das variáveis — XGBoost")

    feature_importance = load_result_image(
        "feature_importance.png"
    )

    if feature_importance is not None:

        st.image(
            feature_importance,
            width="stretch",
        )

    else:

        st.warning("Imagem de importância das variáveis não encontrada.")

    st.divider()

    st.subheader("Explicação de uma previsão — SHAP")

    shap_local = load_result_image("shap_local.png")

    if shap_local is not None:

        st.image(
            shap_local,
            width="stretch",
        )

    else:

        st.warning("Imagem SHAP local não encontrada.")

    st.divider()

    st.subheader("Curva ROC — XGBoost")

    roc_curve = load_result_image(
        "XGBClassifier_roc.png"
    )

    if roc_curve is not None:

        st.image(
            roc_curve,
            width="stretch",
        )

    else:

        st.warning("Imagem da curva ROC não encontrada.")


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "FRAUD DETECTION ANALYTICS | "
    "MACHINE LEARNING | DATA ANALYTICS | RISK INTELLIGENCE"
)
