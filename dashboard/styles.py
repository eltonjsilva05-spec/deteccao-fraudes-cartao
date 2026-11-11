
def apply_styles(st):
    st.markdown(
        """
        <style>

        /* =========================================
           IDENTIDADE VISUAL — FRAUD ANALYTICS
        ========================================= */

        :root {
            --background: #0B0E13;
            --surface: #11151C;
            --surface-hover: #171D27;
            --border: #263140;
            --text-primary: #F1F5F9;
            --text-secondary: #94A3B8;
            --blue: #3B82F6;
            --blue-light: #60A5FA;
            --green: #22C55E;
            --red: #EF4444;
            --cyan: #06B6D4;
        }

        /* =========================================
           FUNDO E CONTEÚDO
        ========================================= */

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 0%,
                    rgba(37, 99, 235, 0.10),
                    transparent 35%
                ),
                var(--background);

            color: var(--text-primary);
        }

        .block-container {
            max-width: 1550px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* =========================================
           TIPOGRAFIA
        ========================================= */

        h1, h2, h3, h4 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        p, label, li {
            color: #CBD5E1;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--text-secondary);
        }

        /* =========================================
           BARRA LATERAL
        ========================================= */

        [data-testid="stSidebar"] {
            background: #0E131B;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-primary) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--border);
        }

        /* =========================================
           CARTÕES DE MÉTRICAS
        ========================================= */

        [data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #171E29,
                #11151C
            );

            border: 1px solid #293548;
            border-radius: 14px;
            padding: 20px 22px;

            box-shadow:
                0 8px 24px rgba(0, 0, 0, 0.20);

            transition:
                border-color 0.2s ease,
                transform 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--blue);
            transform: translateY(-2px);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-size: 27px !important;
            font-weight: 750 !important;
        }

        [data-testid="stMetricDelta"] {
            font-weight: 600;
        }

        /* =========================================
           ABAS DE NAVEGAÇÃO
        ========================================= */

        button[data-baseweb="tab"] {
            color: var(--text-secondary);
            font-weight: 600;
            padding: 13px 18px;
            border-bottom: 2px solid transparent;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--blue-light);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--blue-light);
            border-bottom-color: var(--blue);
        }

        /* =========================================
           GRÁFICOS E TABELAS
        ========================================= */

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px;
        }

        [data-testid="stImage"] img {
            border-radius: 12px;
        }

        /* =========================================
           CABEÇALHOS E CONTAINERS
        ========================================= */

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border) !important;
            border-radius: 16px;
        }

        /* =========================================
           BOTÕES
        ========================================= */

        .stButton > button {
            background: linear-gradient(
                100deg,
                #2563EB,
                #0891B2
            );

            color: #FFFFFF;
            border: 1px solid transparent;
            border-radius: 9px;
            font-weight: 600;

            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            border-color: var(--blue-light);
            box-shadow:
                0 5px 18px rgba(59, 130, 246, 0.25);

            color: #FFFFFF;
        }

        /* =========================================
           SELETORES
        ========================================= */

        [data-baseweb="select"] > div {
            background-color: var(--surface);
            border-color: var(--border);
            color: var(--text-primary);
        }

        /* =========================================
           ALERTAS
        ========================================= */

        [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        /* =========================================
           DIVISORES
        ========================================= */

        hr {
            border-color: var(--border);
        }

        /* =========================================
           RODAPÉ
        ========================================= */

        footer {
            visibility: hidden;
        }

        /* =========================================
           RESPONSIVIDADE
        ========================================= */

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            [data-testid="stMetric"] {
                padding: 15px;
            }

            [data-testid="stMetricValue"] {
                font-size: 22px !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
