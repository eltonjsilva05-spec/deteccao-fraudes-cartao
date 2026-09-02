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

O objetivo principal foi desenvolver uma solução **reprodutível, modular e orientada à análise de resultados**.

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

O dataset bruto não é versionado no GitHub. Caso não esteja presente localmente, o pipeline realiza o download automaticamente.

---

## 🏗️ Pipeline

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Preprocessing
 └── criação de Amount_log
   │
   ▼
Train / Test Split
   │
   ├──────────────────┐
   ▼                  ▼
StandardScaler       SMOTE
 Train only          Train only
   │                  │
   └────────┬─────────┘
            ▼
      Model Training
            │
      ┌─────┼──────────────┐
      ▼     ▼              ▼
 Logistic  Random Forest  XGBoost
Regression
      │     │              │
      └─────┴──────────────┘
            │
            ▼
     Model Evaluation
            │
            ├── Precision
            ├── Recall
            ├── F1-Score
            ├── ROC-AUC
            └── Average Precision
            │
            ▼
   Threshold Optimization
            │
            ▼
    SHAP Explainability
```

---

## 🔬 Pré-processamento

### Criação de `Amount_log`

Durante o pré-processamento, é criada uma versão logarítmica da variável `Amount`:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Essa transformação é mantida no DataFrame como uma feature derivada.

> **Observação:** na versão atual do pipeline, o treinamento utiliza a variável original `Amount`, que é posteriormente normalizada com `StandardScaler`. A feature `Amount_log` é criada durante o pré-processamento, mas ainda não é utilizada pelos modelos.

Essa informação é documentada explicitamente para manter o README alinhado com a implementação atual.

### Normalização

A variável `Amount` é normalizada utilizando `StandardScaler`.

O scaler é ajustado somente sobre o conjunto de treinamento:

```text
Treino → fit_transform()
Teste  → transform()
```

Isso evita **data leakage**, impedindo que informações estatísticas do conjunto de teste influenciem o treinamento.

O scaler utilizado no projeto também é salvo em:

```text
models/scaler.pkl
```

### Tratamento do desbalanceamento

O **SMOTE** é aplicado exclusivamente ao conjunto de treinamento utilizado pelos modelos de árvore.

```text
Treino original
      ↓
StandardScaler
      ↓
SMOTE
      ↓
Treino balanceado
```

A Logistic Regression utiliza `class_weight="balanced"` em vez de SMOTE.

O conjunto de teste mantém sua distribuição original, proporcionando uma avaliação mais próxima do cenário real.

---

# 🤖 Modelos avaliados

Foram comparados três modelos:

### Logistic Regression

Utilizado como **baseline**, com `class_weight="balanced"`.

### Random Forest

Modelo baseado em múltiplas árvores de decisão, treinado utilizando o conjunto balanceado pelo SMOTE.

### XGBoost

Modelo baseado em Gradient Boosting e utilizado como principal candidato para a solução final.

---

# 📈 Resultados

A avaliação foi realizada sobre **85.443 transações de teste**, mantendo o desbalanceamento original.

| Modelo              |  Precision |     Recall |   F1-Score |    ROC-AUC | Average Precision |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ----------------: |
| Logistic Regression |     0.0654 | **0.8784** |     0.1217 |     0.9676 |            0.7033 |
| Random Forest       |     0.6139 |     0.8378 |     0.7086 | **0.9790** |            0.7870 |
| **XGBoost**         | **0.8521** |     0.8176 | **0.8345** |     0.9757 |        **0.8405** |

### 🏆 Modelo selecionado

O **XGBoost apresentou o melhor F1-Score** entre os modelos avaliados.

```text
Precision:           0.8521
Recall:              0.8176
F1-Score:            0.8345
ROC-AUC:             0.9757
Average Precision:   0.8405
```

Na avaliação com threshold `0.50`:

```text
Verdadeiros Negativos: 85.274
Falsos Positivos:          21
Falsos Negativos:          27
Verdadeiros Positivos:    121
```

---

# 🎚️ Otimização do Threshold

O threshold padrão `0.50` não necessariamente representa a melhor decisão para um problema de fraude.

Por isso, foram avaliados thresholds entre `0.01` e `0.99`, utilizando o F1-Score como critério de seleção.

### Melhor resultado encontrado

```text
Threshold: 0.77

Precision: 0.8947
Recall:    0.8041
F1-Score:  0.8470
```

| Configuração       |  Precision | Recall |   F1-Score |
| ------------------ | ---------: | -----: | ---------: |
| Threshold 0.50     |     0.8521 | 0.8176 |     0.8345 |
| **Threshold 0.77** | **0.8947** | 0.8041 | **0.8470** |

O aumento do threshold tornou o modelo mais seletivo ao classificar uma transação como fraude.

Neste experimento:

* Precision aumentou;
* Recall apresentou uma pequena redução;
* F1-Score aumentou.

> **Importante:** o threshold `0.77` é específico deste experimento e deste conjunto de teste. Em produção, sua definição deveria considerar custos reais de falsos positivos e falsos negativos, validação temporal e monitoramento do modelo.

![Threshold Analysis](results/threshold_analysis.png)

---

# 🔎 Explainability com SHAP

Além de identificar possíveis fraudes, é importante entender **por que o modelo tomou determinada decisão**.

O projeto utiliza **SHAP (SHapley Additive exPlanations)** para gerar explicações locais das previsões.

Isso permite analisar a contribuição das variáveis para a classificação de uma determinada transação.

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

# 🧠 Principais aprendizados

Durante o desenvolvimento foram praticados conceitos importantes de **Dados, Python e Machine Learning**:

* análise e manipulação de dados com Pandas;
* tratamento de datasets altamente desbalanceados;
* divisão estratificada de treino e teste;
* prevenção de data leakage;
* transformação de features;
* normalização de dados;
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

### 1. Clonar o repositório

```bash
git clone https://github.com/eltonjsilva05-spec/deteccao-fraudes-cartao.git

cd deteccao-fraudes-cartao
```

### 2. Criar o ambiente virtual

**Windows:**

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o pipeline

```bash
python src/main.py
```

Se o dataset não estiver presente em `data/raw/`, o projeto fará o download automaticamente.

Os modelos e resultados serão gerados nas pastas correspondentes.

---

# 🚀 Próximas evoluções

Possíveis melhorias para uma versão futura:

* utilizar `Amount_log` efetivamente no treinamento e avaliar seu impacto;
* validação temporal (*time-based validation*);
* engenharia de features;
* calibração das probabilidades;
* otimização de hiperparâmetros;
* comparação com modelos adicionais;
* análise baseada em custo financeiro de falsos positivos e falsos negativos;
* criação de pipeline de inferência para novas transações;
* disponibilização do modelo através de API;
* monitoramento de *data drift* e *model drift*;
* containerização com Docker;
* integração com pipelines de dados.

---

# 👨‍💻 Sobre

Este projeto faz parte do meu portfólio de desenvolvimento profissional na área de **Dados e Tecnologia**, com foco em:

**Python • SQL • ETL/ELT • Machine Learning • Análise de Dados • Power BI**

O objetivo é desenvolver soluções orientadas a dados capazes de transformar informações em **insights, automação e suporte à tomada de decisão**.

### Elton Jhon Silva

**Data & Python | SQL | ETL/ELT | Machine Learning | Power BI**

🔗 [LinkedIn](https://www.linkedin.com/in/eltonjsilva)

🔗 [GitHub](https://github.com/eltonjsilva05-spec)

---

⭐ Se este projeto foi útil ou interessante, considere deixar uma estrela no repositório.
