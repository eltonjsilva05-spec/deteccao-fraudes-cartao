# 🛡️ Detecção de Fraudes em Cartões de Crédito com Machine Learning

Projeto de Machine Learning desenvolvido em Python para identificar transações potencialmente fraudulentas em cartões de crédito.

O projeto implementa um pipeline de análise de dados, pré-processamento, tratamento de classes desbalanceadas, treinamento e comparação de modelos preditivos, avaliação de métricas, análise de threshold e explicabilidade com SHAP.

Também conta com um dashboard interativo desenvolvido com Streamlit e Plotly, permitindo explorar os resultados e alternar entre diferentes modelos de Machine Learning.

## 🚀 Dashboard interativo

**[Acessar o Fraud Detection Analytics](https://deteccao-fraudes-cartao-gu2sks9hkbc4l2iw3nzeap.streamlit.app/)**

O dashboard permite explorar os resultados do projeto por meio de quatro áreas:

- **Visão Geral:** indicadores e resumo dos resultados.
- **Modelos:** comparação entre Logistic Regression, Random Forest e XGBoost, com seletor para alternar o modelo analisado.
- **Threshold:** análise do impacto do limiar de classificação sobre Precision, Recall e F1-Score.
- **Explicabilidade:** visualização da importância das variáveis e interpretação das previsões com SHAP.

## 🎯 Objetivo do projeto

Desenvolver um fluxo de análise e classificação de transações financeiras, buscando identificar fraudes e compreender o equilíbrio entre a detecção de operações fraudulentas e a redução de falsos positivos.

O projeto contempla:

- Análise exploratória de dados.
- Pré-processamento e transformação de variáveis.
- Tratamento do desbalanceamento entre classes.
- Treinamento de modelos de classificação.
- Comparação de desempenho.
- Avaliação de métricas.
- Análise de diferentes thresholds.
- Explicabilidade das previsões.
- Desenvolvimento de dashboard interativo.

## 📊 Dataset

Foi utilizado o conjunto de dados público **Credit Card Fraud Detection**.

| Característica | Quantidade |
|---|---:|
| Total de transações | 284.807 |
| Variáveis | 31 |
| Transações fraudulentas | 492 |
| Transações legítimas | 284.315 |

A variável `Class` identifica o tipo de transação:

- `0`: transação legítima.
- `1`: transação fraudulenta.

As variáveis `V1` até `V28` correspondem a componentes transformados por PCA. O conjunto também contém as variáveis `Time` e `Amount`.

O dataset original não é versionado no repositório. O pipeline realiza o carregamento dos dados quando necessário.

## ⚙️ Pipeline de Machine Learning

O desenvolvimento foi organizado nas seguintes etapas:

1. Carregamento dos dados.
2. Análise exploratória.
3. Verificação da distribuição das classes.
4. Pré-processamento das variáveis.
5. Separação dos dados de treinamento e teste.
6. Aplicação de técnicas para lidar com o desbalanceamento.
7. Treinamento dos modelos.
8. Avaliação e comparação dos resultados.
9. Análise de threshold.
10. Explicabilidade das previsões.

### Pré-processamento

Foram implementadas as seguintes etapas:

- Criação da variável `Amount_log` por meio de transformação logarítmica com `np.log1p()`.
- Separação dos dados em conjuntos de treinamento e teste.
- Aplicação do `StandardScaler` à variável `Amount`.
- Ajuste do scaler exclusivamente nos dados de treinamento, evitando data leakage.
- Criação da variável `Amount_scaled`.
- Remoção da variável `Amount` original após a normalização.
