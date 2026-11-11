
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# IDENTIDADE VISUAL
# =========================================================

BACKGROUND = "#0b1120"
PANEL = "#111827"
GRID = "#263449"

TEXT = "#e2e8f0"
MUTED = "#94a3b8"

CYAN = "#38bdf8"
BLUE = "#3b82f6"
PURPLE = "#a78bfa"
GREEN = "#34d399"
YELLOW = "#fbbf24"
RED = "#f87171"

COLORS = {
    "Precision": CYAN,
    "Recall": PURPLE,
    "F1": GREEN,
    "ROC_AUC": YELLOW,
    "Average_Precision": BLUE,
}


# =========================================================
# CONFIGURAÇÃO PADRÃO DOS GRÁFICOS
# =========================================================

def aplicar_layout(fig, height=450, title=None):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 18,
                "color": TEXT
            }
        } if title else None,

        height=height,

        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,

        font=dict(
            family="Arial, sans-serif",
            color=TEXT,
            size=12
        ),

        margin=dict(
            l=45,
            r=30,
            t=75,
            b=45
        ),

        legend=dict(
            title="Indicadores",
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(color=TEXT)
        ),

        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=CYAN,
            font=dict(
                color=TEXT,
                size=12
            )
        ),

        xaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            zerolinecolor=GRID,
            tickfont=dict(color=MUTED),
            title_font=dict(color=TEXT)
        ),

        yaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            zerolinecolor=GRID,
            tickfont=dict(color=MUTED),
            title_font=dict(color=TEXT)
        )
    )

    return fig


# =========================================================
# COMPARAÇÃO DOS MODELOS
# =========================================================

def model_comparison_chart(metrics):

    columns = [
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]

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
        color_discrete_map=COLORS,
        labels={
            "Modelo": "Modelo",
            "Valor": "Resultado",
            "Indicador": "Indicador"
        },
        text_auto=".1%"
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Indicador: %{fullData.name}<br>"
            "Resultado: %{y:.2%}"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1.12],
        title="Desempenho",
        showgrid=True
    )

    fig.update_xaxes(
        title="Modelos"
    )

    return aplicar_layout(
        fig,
        height=470,
        title="Comparação de desempenho dos modelos"
    )


# =========================================================
# COMPARAÇÃO 3D
# =========================================================

def model_comparison_3d_chart(metrics):

    indicators = [
        ("Precision", "Precision", CYAN),
        ("Recall", "Recall", PURPLE),
        ("F1", "F1-Score", GREEN),
        ("ROC_AUC", "ROC-AUC", YELLOW)
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
        title=dict(
            text="Comparação 3D dos modelos de Machine Learning",
            x=0.5,
            xanchor="center",
            font=dict(
                size=18,
                color=TEXT
            )
        ),

        height=650,

        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,

        font=dict(
            color=TEXT,
            size=12
        ),

        legend=dict(
            title="Indicadores",
            bgcolor="rgba(17, 24, 39, 0.8)"
        ),

        scene=dict(
            bgcolor=BACKGROUND,

            xaxis=dict(
                title="Modelos",
                gridcolor=GRID,
                backgroundcolor=PANEL,
                color=MUTED
            ),

            yaxis=dict(
                title="Indicadores",
                gridcolor=GRID,
                backgroundcolor=PANEL,
                color=MUTED
            ),

            zaxis=dict(
                title="Desempenho",
                tickformat=".0%",
                range=[0, 1],
                gridcolor=GRID,
                backgroundcolor=PANEL,
                color=MUTED
            ),

            camera=dict(
                eye=dict(
                    x=1.7,
                    y=1.7,
                    z=1.2
                )
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=1.4,
                y=1.2,
                z=0.9
            )
        ),

        margin=dict(
            l=0,
            r=0,
            t=75,
            b=0
        )
    )

    return fig


# =========================================================
# ANÁLISE DE THRESHOLD
# =========================================================

def threshold_chart(threshold):

    fig = go.Figure()

    indicators = [
        ("Precision", "Precisão", CYAN),
        ("Recall", "Recall", PURPLE),
        ("F1", "F1-Score", GREEN)
    ]

    for column, label, color in indicators:

        fig.add_trace(
            go.Scatter(
                x=threshold["Threshold"],
                y=threshold[column],

                mode="lines",

                name=label,

                line=dict(
                    color=color,
                    width=3,
                    shape="spline"
                ),

                hovertemplate=(
                    "<b>Threshold:</b> %{x:.2f}<br>"
                    "<b>Resultado:</b> %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_xaxes(
        title="Threshold",
        tickformat=".2f"
    )

    fig.update_yaxes(
        title="Resultado",
        tickformat=".0%",
        range=[0, 1]
    )

    fig.update_layout(
        hovermode="x unified"
    )

    return aplicar_layout(
        fig,
        height=470,
        title="Impacto do limiar de classificação"
    )


# =========================================================
# AVERAGE PRECISION
# =========================================================

def average_precision_chart(metrics):

    fig = px.bar(
        metrics,
        x="Modelo",
        y="Average_Precision",
        color="Modelo",

        color_discrete_sequence=[
            CYAN,
            PURPLE,
            GREEN
        ],

        text_auto=".2%",

        labels={
            "Modelo": "Modelo",
            "Average_Precision": "Average Precision"
        }
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,

        hovertemplate=(
            "<b>%{x}</b><br>"
            "Average Precision: %{y:.2%}"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(
        title="Average Precision",
        tickformat=".0%",
        range=[0, 1.12]
    )

    fig.update_xaxes(
        title="Modelos"
    )

    fig.update_layout(
        showlegend=False
    )

    return aplicar_layout(
        fig,
        height=400,
        title="Average Precision por modelo"
    )
