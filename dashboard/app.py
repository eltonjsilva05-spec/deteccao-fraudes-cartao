
from pathlib import Path

import streamlit as st
import pandas as pd

from metrics import load_metrics, load_threshold, load_result_image, get_best_model
from charts import (
    model_comparison_chart,
    model_comparison_3d_chart,
    threshold_chart,
    average_precision_chart,
)
from styles import apply_styles


# ==================================================
# CONFIGURAÇÃO
# ==================================================

st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

apply_styles(st)


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================

try:
    metrics = load_metrics()
    threshold_data = load_threshold()

    if metrics is None or metrics.empty:
        st.error("Não foi possível carregar os resultados dos modelos.")
        st.stop()

except Exception as error:
    st.error(f"Erro ao carregar os dados: {error}")
    st.stop()


# ==================================================
# BARRA LATERAL
# ==================================================

with st.sidebar:
    st.title("🛡️ Fraud Analytics")

    st.caption("MACHINE LEARNING | DATA ANALYTICS")

    st.divider()

    st.markdown("### Navegação")
    st.write("Utilize as abas do painel para explorar os resultados.")

    st.divider()

    st.markdown("### Tecnologias")

    st.markdown(
        """
        - Python
        - Pandas
        - Scikit-learn
        - XGBoost
        - Streamlit
        - Plotly
        - SHAP
        """
    )

    st.divider()

    st.caption("Projeto de portfólio — Detecção de Fraudes")


# ==================================================
# CABEÇALHO PRINCIPAL
# ==================================================

with st.container(border=True):

    st.markdown("### 🛡️ MACHINE LEARNING ANALYTICS")

    st.title("Fraud Detection Analytics")

    st.write(
        "Inteligência analítica aplicada à identificação "
        "e avaliação de modelos de detecção de fraudes "
        "em cartões de crédito."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("PYTHON")

    with col2:
        st.info("MACHINE LEARNING")

    with col3:
        st.info("DATA ANALYTICS")


# ==================================================
# INDICADORES PRINCIPAIS
# ==================================================

st.markdown("## Indicadores dos Modelos")

best_model = get_best_model(metrics)

if best_model is not None:
    best_name = str(best_model["Modelo"])
    best_f1 = float(best_model["F1"])
    best_precision = float(best_model["Precision"])
    best_recall = float(best_model["Recall"])
    best_auc = float(best_model["ROC_AUC"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Melhor modelo", best_name)

    with col2:
        st.metric("F1-Score", f"{best_f1:.4f}")

    with col3:
        st.metric("Precision", f"{best_precision:.4f}")

    with col4:
        st.metric("Recall", f"{best_recall:.4f}")

else:
    st.warning("Não foi possível identificar o melhor modelo.")


# ==================================================
# ABAS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visão Geral",
        "🤖 Modelos",
        "🎯 Threshold",
        "🔍 Explicabilidade",
    ]
)


# ==================================================
# ABA 1 — VISÃO GERAL
# ==================================================

with tab1:

    st.subheader("Visão Geral do Desempenho")

    st.write(
        "Comparação dos principais indicadores dos modelos "
        "utilizados na detecção de fraudes."
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

    st.markdown("### Resultados dos Modelos")

    st.dataframe(
        metrics,
        width="stretch",
        hide_index=True,
    )


# ==================================================
# ABA 2 — MODELOS
# ==================================================

with tab2:

    st.subheader("Comparação entre Modelos")

    st.write(
        "Análise comparativa dos algoritmos Logistic Regression, "
        "Random Forest e XGBoost."
    )

    st.plotly_chart(
        model_comparison_3d_chart(metrics),
        width="stretch",
        key="models_comparison_3d",
    )

    st.markdown("### Matriz de Confusão — XGBoost")

    confusion_matrix = load_result_image(
        "XGBClassifier_confusion_matrix.png"
    )

    if confusion_matrix is not None:
        st.image(confusion_matrix, width="stretch")
    else:
        st.warning("Imagem da matriz de confusão não encontrada.")


# ==================================================
# ABA 3 — THRESHOLD
# ==================================================

with tab3:

    st.subheader("Análise de Threshold")

    st.write(
        "Ajustar o threshold permite avaliar o equilíbrio entre "
        "Precision e Recall na identificação de transações fraudulentas."
    )

    if threshold_data is not None and not threshold_data.empty:

        st.plotly_chart(
            threshold_chart(threshold_data),
            width="stretch",
            key="threshold_analysis",
        )

        st.markdown("### Dados da Análise")

        st.dataframe(
            threshold_data,
            width="stretch",
            hide_index=True,
        )

    else:
        st.warning("Dados de análise de threshold não encontrados.")

    st.markdown("### Curva Precision-Recall")

    precision_recall = load_result_image(
        "XGBClassifier_precision_recall.png"
    )

    if precision_recall is not None:
        st.image(precision_recall, width="stretch")
    else:
        st.warning("Imagem Precision-Recall não encontrada.")


# ==================================================
# ABA 4 — EXPLICABILIDADE
# ==================================================

with tab4:

    st.subheader("Explicabilidade do Modelo")

    st.write(
        "Visualização das variáveis mais relevantes para as "
        "previsões do modelo de detecção de fraudes."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Importância das Variáveis")

        feature_importance = load_result_image(
            "feature_importance.png"
        )

        if feature_importance is not None:
            st.image(feature_importance, width="stretch")
        else:
            st.warning("Imagem de importância das variáveis não encontrada.")

    with col2:

        st.markdown("### Análise SHAP")

        shap_image = load_result_image("shap_local.png")

        if shap_image is not None:
            st.image(shap_image, width="stretch")
        else:
            st.warning("Imagem SHAP não encontrada.")

    st.markdown("### Curva ROC")

    roc_image = load_result_image("XGBClassifier_roc.png")

    if roc_image is not None:
        st.image(roc_image, width="stretch")
    else:
        st.warning("Imagem ROC não encontrada.")


# ==================================================
# RODAPÉ
# ==================================================

st.divider()

st.caption(
    "Fraud Detection Analytics | Projeto de Machine Learning "
    "e Análise de Dados"
)
