
def apply_styles(st):
    st.markdown(
        """
        <style>

        /* FUNDO GERAL */
        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    #172554 0%,
                    #0b1120 40%,
                    #070d19 100%
                );
            color: #e2e8f0;
        }

        /* CONTEUDO PRINCIPAL */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* BARRA LATERAL */
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }

        /* TITULOS */
        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 700;
            letter-spacing: -0.4px;
        }

        p, label {
            color: #cbd5e1;
        }

        /* CARTOES DE METRICAS */
        [data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #172554,
                #111827
            );
            border: 1px solid #2563eb;
            padding: 22px;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
            transition: all 0.25s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: #38bdf8;
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(14, 165, 233, 0.12);
        }

        [data-testid="stMetricLabel"] {
            color: #93c5fd !important;
            font-size: 14px;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-size: 27px;
            font-weight: 800;
        }

        /* ABAS */
        button[data-baseweb="tab"] {
            color: #94a3b8;
            font-weight: 600;
            padding: 12px 18px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8;
        }

        /* GRAFICOS E TABELAS */
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 10px;
        }

        /* CABECALHO */
        .hero-card {
            background: linear-gradient(
                120deg,
                #172554 0%,
                #1e1b4b 55%,
                #0f172a 100%
            );
            border: 1px solid #2563eb;
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 28px;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
        }

        .hero-card h2 {
            color: #f8fafc;
            font-size: 30px;
            margin-bottom: 10px;
        }

        .hero-card p {
            color: #bfdbfe;
            font-size: 15px;
            margin-bottom: 0;
        }

        /* DESCRICOES */
        .section-description {
            color: #94a3b8;
            font-size: 14px;
        }

        /* LINHAS DIVISORIAS */
        hr {
            border-color: #263449;
        }

        /* BOTOES */
        .stButton > button {
            background: linear-gradient(
                90deg,
                #2563eb,
                #0891b2
            );
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            border: none;
            box-shadow: 0 5px 18px rgba(14, 165, 233, 0.3);
            color: white;
        }

        /* SELETORES */
        [data-baseweb="select"] > div {
            background-color: #111827;
            border-color: #334155;
            color: #e2e8f0;
        }

        /* RODAPE */
        footer {
            visibility: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
