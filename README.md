# 💳 Detecção de Fraudes em Cartões de Crédito com Machine Learning

> Pipeline de Machine Learning desenvolvido em Python para identificação de transações potencialmente fraudulentas, com tratamento de desbalanceamento, comparação de modelos, otimização de threshold e explicabilidade com SHAP.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189E68)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)](https://shap.readthedocs.io/)
[![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-SMOTE-orange)](https://imbalanced-learn.org/)

---

## 🎯 Visão geral

Sistemas de detecção de fraude precisam identificar transações suspeitas sem gerar uma quantidade excessiva de falsos positivos.

Este projeto implementa um pipeline completo de Machine Learning para esse cenário, passando por:

* carregamento automatizado dos dados;
* pré-processamento;
* divisão estratificada entre treino e teste;
* normalização;
* tratamento de desbalanceamento com SMOTE;
* treinamento de diferentes modelos;
* avaliação com métricas apropriadas para fraude;
* otimização do threshold de decisão;
* explicabilidade utilizando SHAP;
* persistência dos modelos e resultados.

O objetivo principal é desenvolver uma solução **reprodutível, modular e orientada à análise de resultados**.

---

## 📊 Dataset

Foi utilizado o dataset público **Credit Card Fraud Detection**, disponibilizado pelo TensorFlow.

| Característica  |       Valor |
| --------------- | ----------: |
| Transações      | **284.807** |
| Variáveis       |      **31** |
| Fraudes         |     **492** |
| Classe legítima |         `0` |
| Classe fraude   |         `1` |

O dataset apresenta um forte desbalanceamento entre as classes.

As variáveis `V1` até `V28` representam componentes transformados por PCA. O conjunto também contém as variáveis `Time` e `Amount`.

O dataset bruto **não é versionado no GitHub**. Caso não esteja presente localmente, o pipeline realiza o download automaticamente.

---

# 🏗️ Pipeline do projeto

```text
                 ┌─────────────────────┐
                 │       Dataset       │
                 │  Credit Card Fraud  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Data Loading     │
                 │       Pandas       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Preprocessing    │
                 │ criação Amount_log  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Train / Test Split │
                 │      Stratified     │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ StandardScaler│    │    SMOTE     │
          │  Train only   │    │  Train only  │
          └───────┬───────┘    └───────┬──────┘
                  │                    │
                  └─────────┬──────────┘
                            ▼
                    Model Training
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       Logistic        Random Forest   XGBoost
       Regression
              │             │             │
              └─────────────┴─────────────┘
                            │
                            ▼
                   Model Evaluation
                            │
                   ┌────────┼────────┐
                   ▼        ▼        ▼
               Precision  Recall  F1-Score
                   │        │        │
                   └────────┼────────┘
                            │
                            ▼
                       ROC-AUC / AP
                            │
                            ▼
                  Threshold Optimization
                            │
                            ▼
                    SHAP Explainability
```

---

# 🔬 Pré-processamento

## Criação de `Amount_log`

Durante o pré-processamento, é criada uma versão logarítmica da variável `Amount`:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Essa transformação reduz o impacto de valores muito discrepantes e cria uma feature derivada que pode ser utilizada em futuras versões do pipeline.

> **Observação:** na versão atual do pipeline, o treinamento utiliza a variável original `Amount`, que posteriormente é normalizada com `StandardScaler`. A feature `Amount_log` é criada durante o pré-processamento, mas ainda não é utilizada pelos modelos.

Essa informação é documentada explicitamente para manter o README alinhado com a implementação atual.

## Normalização

A variável `Amount` é normalizada utilizando `StandardScaler`.

O scaler é ajustado **somente sobre o conjunto de treinamento**:

```text
Treino → fit_transform()

Teste  → transform()
```

Isso evita **data leakage**, impedindo que informações estatísticas do conjunto de teste influenciem o treinamento.

O scaler utilizado no projeto também é salvo em:

```text
models/scaler.pkl
```

## Tratamento do desbalanceamento

O dataset possui uma quantidade muito pequena de fraudes em comparação com transações legítimas.

Para os modelos de árvore, o **SMOTE** é aplicado exclusivamente ao conjunto de treinamento:

```text
Treino original
      ↓
StandardScaler
      ↓
SMOTE
      ↓
Treino balanceado
```

A **Logistic Regression** utiliza `class_weight="balanced"` em vez de SMOTE.

O conjunto de teste mantém sua distribuição original, proporcionando uma avaliação mais próxima do cenário real.

---

# 🤖 Modelos avaliados

Foram comparados três modelos:

## Logistic Regression

Utilizada como modelo de referência (*baseline*), com:

```python
class_weight="balanced"
```

## Random Forest

Modelo baseado em múltiplas árvores de decisão, treinado utilizando o conjunto balanceado pelo SMOTE.

## XGBoost

Modelo baseado em Gradient Boosting e utilizado como principal candidato para a solução final.

---

# 📈 Resultados

A avaliação foi realizada sobre **85.443 transações de teste**, mantendo o desbalanceamento original.

| Modelo              |  Precision | Recall |   F1-Score |    ROC-AUC | Average Precision |
| ------------------- | ---------: | -----: | ---------: | ---------: | ----------------: |
| Logistic Regression |     0.0654 | 0.8784 |     0.1217 |     0.9676 |            0.7033 |
| Random Forest       |     0.6139 | 0.8378 |     0.7086 | **0.9790** |            0.7870 |
| **XGBoost**         | **0.8521** | 0.8176 | **0.8345** |     0.9757 |        **0.8405** |

## 🏆 Modelo selecionado

O **XGBoost apresentou o melhor F1-Score** entre os modelos avaliados.

```text
Precision:           0.8521
Recall:              0.8176
F1-Score:            0.8345
ROC-AUC:             0.9757
Average Precision:   0.8405
```

Na avaliação padrão com threshold `0.50`:

```text
Verdadeiros Negativos: 85.274
Falsos Positivos:          21
Falsos Negativos:          27
Verdadeiros Positivos:    121
```

---

# 🎚️ Otimização do Threshold

Em problemas de fraude, o threshold padrão `0.50` nem sempre representa a melhor decisão de negócio.

Por isso, foram avaliados thresholds entre `0.01` e `0.99`, utilizando o F1-Score como critério de seleção.

## Melhor resultado encontrado

```text
Threshold: 0.77
Precision: 0.8947
Recall:    0.8041
F1-Score:  0.8470
```

### Comparação

| Configuração       |  Precision | Recall |   F1-Score |
| ------------------ | ---------: | -----: | ---------: |
| Threshold 0.50     |     0.8521 | 0.8176 |     0.8345 |
| **Threshold 0.77** | **0.8947** | 0.8041 | **0.8470** |

### Interpretação

Ao aumentar o threshold, o modelo se torna mais seletivo ao classificar uma transação como fraude.

Neste experimento, isso resultou em:

* aumento da Precision;
* pequena redução do Recall;
* aumento do F1-Score.

> **Importante:** o threshold `0.77` é específico deste experimento e deste conjunto de teste. Em um ambiente de produção, sua definição deveria considerar custos reais de falsos positivos e falsos negativos, validação temporal e monitoramento do modelo.

![Threshold Analysis](results/threshold_analysis.png)

---

# 🔎 Explainability com SHAP

Além de prever fraudes, o projeto busca responder:

> **Por que o modelo classificou determinada transação como potencialmente fraudulenta?**

Para isso foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**.

O projeto gera uma explicação local para uma transação específica, permitindo analisar a contribuição das variáveis para a decisão do modelo.

### Exemplo gerado pelo projeto

![SHAP](results/shap_local.png)

---

# 📊 Visualizações

### Matriz de Confusão

![Confusion Matrix](results/XGBClassifier_confusion_matrix.png)

### Curva ROC

![ROC Curve](results/XGBClassifier_roc.png)

### Precision-Recall

![Precision Recall](results/XGBClassifier_precision_recall.png)

### Importância das Variáveis

![Feature Importance](results/feature_importance.png)

---

# 🧠 Principais aprendizados técnicos

Durante o desenvolvimento foram praticados conceitos importantes de **Dados, Python e Machine Learning**, incluindo:

* análise e manipulação de dados com Pandas;
* tratamento de datasets altamente desbalanceados;
* divisão estratificada de treino e teste;
* prevenção de data leakage;
* transformação de features;
* normalização de dados;
* transformação logarítmica;
* SMOTE;
* `class_weight`;
* treinamento de modelos supervisionados;
* comparação de algoritmos;
* Precision, Recall e F1-Score;
* ROC-AUC;
* Average Precision;
* matriz de confusão;
* otimização de threshold;
* interpretação de modelos com SHAP;
* persistência de modelos com Joblib;
* organização modular de projetos Python;
* geração automatizada de métricas e visualizações.

---

# 🛠️ Tecnologias

| Tecnologia           | Utilização                      |
| -------------------- | ------------------------------- |
| **Python**           | Linguagem principal             |
| **Pandas**           | Manipulação e análise dos dados |
| **NumPy**            | Operações numéricas             |
| **Scikit-learn**     | Machine Learning e métricas     |
| **XGBoost**          | Gradient Boosting               |
| **Imbalanced-learn** | SMOTE                           |
| **SHAP**             | Explainability                  |
| **Matplotlib**       | Visualização                    |
| **Joblib**           | Persistência dos modelos        |
| **Git / GitHub**     | Versionamento                   |

---

# 📁 Estrutura do projeto

```text
deteccao-fraudes-cartao/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── imagens/
│   └── apresentacao/
│
├── models/
│
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
│
├── results/
│   ├── XGBClassifier_confusion_matrix.png
│   ├── XGBClassifier_precision_recall.png
│   ├── XGBClassifier_roc.png
│   ├── feature_importance.png
│   ├── metricas.csv
│   ├── shap_local.png
│   ├── threshold_analysis.csv
│   └── threshold_analysis.png
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── threshold_analysis.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ▶️ Como executar

## 1. Clonar o repositório

```bash
git clone https://github.com/eltonjsilva05-spec/deteccao-fraudes-cartao.git

cd deteccao-fraudes-cartao
```

## 2. Criar o ambiente virtual

### Windows

```bash
python -m venv .venv
```

### Git Bash

```bash
source .venv/Scripts/activate
```

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## 4. Executar o pipeline

```bash
python src/main.py
```

Caso o dataset ainda não esteja presente em `data/raw/`, o projeto realiza o download automaticamente.

Os resultados serão gerados em:

```text
results/
```

---

# 🚀 Próximas evoluções

Algumas possibilidades para evolução do projeto:

* utilizar `Amount_log` efetivamente no treinamento e avaliar seu impacto;
* validação temporal (*time-based validation*);
* engenharia de features;
* calibração das probabilidades;
* otimização de hiperparâmetros;
* comparação com modelos adicionais;
* análise de custo financeiro de falsos positivos e falsos negativos;
* criação de pipeline de inferência para novas transações;
* API para disponibilização do modelo;
* monitoramento de *data drift* e *model drift*;
* containerização com Docker;
* integração com pipelines de dados.

---

# 👨‍💻 Sobre o projeto

Este projeto faz parte do meu portfólio de transição e desenvolvimento profissional na área de **Dados e Tecnologia**, com foco em:

**Python • SQL • ETL/ELT • Machine Learning • Análise de Dados • Power BI**

Meu objetivo é desenvolver soluções orientadas a dados capazes de transformar informações em **insights, automação e suporte à tomada de decisão**.

## 👤 Autor

**Elton Jhon Silva**

**Data & Python | SQL | ETL/ELT | Machine Learning | Power BI**

🔗 LinkedIn: [linkedin.com/in/eltonjsilva](https://www.linkedin.com/in/eltonjsilva)

🔗 GitHub: [github.com/eltonjsilva05-spec](https://github.com/eltonjsilva05-spec)

---

⭐ Se este projeto foi útil ou interessante, considere deixar uma estrela no repositório.
