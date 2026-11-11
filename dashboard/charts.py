
import plotly.express as px
import plotly.graph_objects as go


def model_comparison_chart(metrics):
    columns = ["Precision", "Recall", "F1", "ROC_AUC"]

    data = metrics.melt(
        id_vars="Modelo",
        value_vars=columns,
        var_name="Indicador",
        value_name="Valor"
    )

    fig = px.bar(
        data,
        x="Modelo",
        y="Valor",
        color="Indicador",
        barmode="group",
        title="Comparação de desempenho dos modelos",
        labels={
            "Modelo": "Modelo",
            "Valor": "Resultado",
            "Indicador": "Indicador"
        }
    )

    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        legend_title="Indicadores",
        height=450
    )

    return fig


def model_comparison_3d_chart(metrics):
    """
    Gráfico 3D interativo para comparar os modelos
    usando Precision, Recall, F1 e ROC-AUC.
    """

    indicators = [
        ("Precision", "Precision", "#38bdf8"),
        ("Recall", "Recall", "#a78bfa"),
        ("F1", "F1-Score", "#34d399"),
        ("ROC_AUC", "ROC-AUC", "#fbbf24")
    ]

    models = metrics["Modelo"].tolist()

    fig = go.Figure()

    for metric_column, metric_label, metric_color in indicators:
        fig.add_trace(
            go.Scatter3d(
                x=models,
                y=[metric_label] * len(models),
                z=metrics[metric_column].tolist(),
                mode="markers",
                name=metric_label,
                marker=dict(
                    size=9,
                    color=metric_color,
                    opacity=0.95,
                    line=dict(
                        color="#f8fafc",
                        width=0.8
                    )
                ),
                hovertemplate=(
                    "<b>Modelo:</b> %{x}<br>"
                    "<b>Indicador:</b> %{y}<br>"
                    "<b>Resultado:</b> %{z:.2%}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title={
            "text": "Comparação 3D dos modelos de Machine Learning",
            "x": 0.5,
            "xanchor": "center"
        },
        height=650,
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font=dict(
            color="#e2e8f0",
            size=12
        ),
        legend=dict(
            title="Indicadores",
            bgcolor="rgba(17, 24, 39, 0.8)"
        ),
        scene=dict(
            bgcolor="#0b1120",
            xaxis=dict(
                title="Modelos",
                gridcolor="#334155",
                backgroundcolor="#111827",
                color="#cbd5e1"
            ),
            yaxis=dict(
                title="Indicadores",
                gridcolor="#334155",
                backgroundcolor="#111827",
                color="#cbd5e1"
            ),
            zaxis=dict(
                title="Desempenho",
                tickformat=".0%",
                range=[0, 1],
                gridcolor="#334155",
                backgroundcolor="#111827",
                color="#cbd5e1"
            ),
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=1.2)
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.2, z=0.9)
        ),
        margin=dict(
            l=0,
            r=0,
            t=70,
            b=0
        )
    )

    return fig


def threshold_chart(threshold):
    fig = go.Figure()

    for column, label in [
        ("Precision", "Precisão"),
        ("Recall", "Recall"),
        ("F1", "F1-Score")
    ]:
        fig.add_trace(
            go.Scatter(
                x=threshold["Threshold"],
                y=threshold[column],
                mode="lines",
                name=label
            )
        )

    fig.update_layout(
        title="Impacto do limiar de classificação",
        xaxis_title="Threshold",
        yaxis_title="Resultado",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified",
        height=450
    )

    return fig


def average_precision_chart(metrics):
    fig = px.bar(
        metrics,
        x="Modelo",
        y="Average_Precision",
        title="Average Precision por modelo",
        text_auto=".2%"
    )

    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=350
    )

    return fig
