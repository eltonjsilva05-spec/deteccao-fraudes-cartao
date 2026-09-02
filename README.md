# 🛡️ Detecção de Fraudes em Cartão de Crédito

Projeto de **Machine Learning para detecção de transações fraudulentas em cartões de crédito**, desenvolvido com Python, Scikit-Learn e XGBoost.

O projeto aborda um dos principais desafios de problemas financeiros de classificação: o **forte desbalanceamento entre transações legítimas e fraudulentas**.

A solução implementa pré-processamento, prevenção de **data leakage**, balanceamento com **SMOTE**, treinamento e comparação de múltiplos modelos, análise de thresholds e explicabilidade utilizando **SHAP**.

---

## 🎯 Objetivo

Desenvolver um modelo capaz de identificar transações potencialmente fraudulentas, buscando um equilíbrio entre:

* **Precision** — reduzir falsos positivos;
* **Recall** — identificar o maior número possível de fraudes;
* **F1-Score** — equilibrar Precision e Recall;
* **Average Precision** — avaliar o desempenho em um cenário altamente desbalanceado.

Além do treinamento, o projeto busca demonstrar um fluxo completo de Machine Learning, desde o carregamento dos dados até a interpretação do modelo final.

---

# 📊 Dataset

O projeto utiliza um dataset público de transações de cartão de crédito disponibilizado pelo TensorFlow.

O dataset é baixado automaticamente na primeira execução e armazenado localmente.

### Características

* **284.807 transações**
* **31 variáveis**
* Apenas **492 transações fraudulentas**
* Forte desbalanceamento entre as classes
* Variável alvo: `Class`

### Classes

```text
0 → Transação legítima
1 → Transação fraudulenta
```

O alto desbalanceamento torna a utilização de métricas como **Precision, Recall, F1-Score e Average Precision** mais relevante do que utilizar apenas Accuracy.

---

# ⚙️ Pipeline de Machine Learning

O projeto segue o seguinte fluxo:

```text
Dataset
   │
   ▼
Carregamento dos dados
   │
   ▼
Pré-processamento
   │
   ├── Transformação logarítmica de Amount
   │
   ▼
Divisão Treino / Teste
   │
   ├── StandardScaler ajustado somente no treino
   │
   ├── SMOTE aplicado somente no treino
   │
   ▼
Treinamento
   │
   ├── Logistic Regression
   ├── Random Forest
   └── XGBoost
   │
   ▼
Avaliação
   │
   ├── Precision
   ├── Recall
   ├── F1-Score
   ├── ROC-AUC
   └── Average Precision
   │
   ▼
Seleção do melhor modelo
   │
   ▼
Análise de Threshold
   │
   ▼
Explicabilidade SHAP
```

---

# 🔒 Prevenção de Data Leakage

Um dos cuidados importantes da implementação foi evitar **data leakage** durante o pré-processamento.

O `StandardScaler` é ajustado exclusivamente utilizando os dados de treinamento:

```python
scaler.fit(X_train)
```

Depois, o mesmo scaler é utilizado para transformar os dados de teste:

```python
scaler.transform(X_test)
```

Da mesma forma, o **SMOTE é aplicado somente ao conjunto de treinamento**.

O conjunto de teste mantém sua distribuição original, permitindo uma avaliação mais próxima do cenário real.

---

# ⚖️ Tratamento do Desbalanceamento

O conjunto de treinamento possui uma diferença extremamente grande entre as classes.

### Distribuição original do treinamento

```text
Classe 0: 199.020
Classe 1:     344
```

Após aplicação do SMOTE:

```text
Classe 0: 199.020
Classe 1: 199.020
```

O teste permanece com a distribuição original:

```text
Classe 0: 85.295
Classe 1:    148
```

Essa abordagem evita utilizar dados artificialmente balanceados para medir o desempenho final do modelo.

---

# 🤖 Modelos Treinados

Foram utilizados três modelos para comparação.

## Logistic Regression

Utilizada como modelo de baseline.

A implementação utiliza `class_weight="balanced"` para lidar com o desbalanceamento sem aplicar SMOTE diretamente nesse modelo.

### Resultado

* Precision: **0.0654**
* Recall: **0.8784**
* F1-Score: **0.1217**
* ROC-AUC: **0.9676**
* Average Precision: **0.7033**

O modelo apresenta alto Recall, mas gera muitos falsos positivos.

---

## Random Forest

Modelo baseado em um conjunto de árvores de decisão.

### Resultado

* Precision: **0.6139**
* Recall: **0.8378**
* F1-Score: **0.7086**
* ROC-AUC: **0.9790**
* Average Precision: **0.7870**

O Random Forest apresentou o maior ROC-AUC entre os modelos avaliados, mas teve desempenho inferior ao XGBoost em F1-Score e Average Precision.

---

## XGBoost

Modelo baseado em Gradient Boosting e utilizado como principal candidato para a solução final.

### Resultado

* Precision: **0.8521**
* Recall: **0.8176**
* F1-Score: **0.8345**
* ROC-AUC: **0.9757**
* Average Precision: **0.8405**

O XGBoost apresentou o melhor equilíbrio geral entre Precision e Recall.

---

# 🏆 Comparação dos Modelos

| Modelo              |  Precision | Recall |   F1-Score |    ROC-AUC | Average Precision |
| ------------------- | ---------: | -----: | ---------: | ---------: | ----------------: |
| Logistic Regression |     0.0654 | 0.8784 |     0.1217 |     0.9676 |            0.7033 |
| Random Forest       |     0.6139 | 0.8378 |     0.7086 | **0.9790** |            0.7870 |
| **XGBoost**         | **0.8521** | 0.8176 | **0.8345** |     0.9757 |        **0.8405** |

### Modelo selecionado

**XGBoost**

O modelo foi selecionado automaticamente com base no **F1-Score**.

```text
F1-Score: 0.8345
```

O modelo final é salvo em:

```text
models/best_model.pkl
```

---

# 🎯 Otimização do Threshold

Além do threshold padrão de `0.50`, foi realizada uma análise de diferentes pontos de decisão entre `0.01` e `0.99`.

O objetivo foi encontrar o threshold que apresentasse o melhor **F1-Score**.

### Threshold padrão

```text
Threshold: 0.50

Precision: 85.21%
Recall:    81.76%
F1-Score:  83.45%
```

### Threshold recomendado

```text
Threshold: 0.77

Precision: 89.47%
Recall:    80.41%
F1-Score:  84.70%
```

Resultado:

* **+4,26 pontos percentuais em Precision**
* **+1,25 ponto percentual em F1-Score**
* redução de **1,35 ponto percentual em Recall**

O threshold `0.77` foi definido como o **threshold recomendado nesta avaliação**, não como um valor universal para produção.

A análise completa é salva em:

```text
results/threshold_analysis.csv
```

e:

```text
results/threshold_analysis.png
```

---

# 🔎 Explicabilidade com SHAP

O projeto também utiliza **SHAP (SHapley Additive exPlanations)** para interpretar as previsões do modelo.

O objetivo é entender quais variáveis mais contribuíram para a classificação de uma transação.

O gráfico de explicabilidade é gerado automaticamente em:

```text
results/shap_local.png
```

Isso permite complementar a avaliação puramente quantitativa com uma análise de **interpretabilidade do modelo**.

---

# 📈 Visualizações

### Curva ROC

![ROC](results/XGBClassifier_roc.png)

---

### Curva Precision-Recall

![Precision Recall](results/XGBClassifier_precision_recall.png)

---

### Matriz de Confusão

![Confusion Matrix](results/XGBClassifier_confusion_matrix.png)

---

### Importância das Variáveis

![Feature Importance](results/feature_importance.png)

---

### Análise de Threshold

![Threshold Analysis](results/threshold_analysis.png)

---

### Explicabilidade SHAP

![SHAP](results/shap_local.png)

---

# 📁 Estrutura do Projeto

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
│   ├── logistic.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── results/
│   ├── metricas.csv
│   ├── threshold_analysis.csv
│   ├── threshold_analysis.png
│   ├── XGBClassifier_roc.png
│   ├── XGBClassifier_precision_recall.png
│   ├── XGBClassifier_confusion_matrix.png
│   ├── feature_importance.png
│   └── shap_local.png
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── threshold_analysis.py
│   ├── explainability.py
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Tecnologias

* Python 3.11
* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn
* XGBoost
* SHAP
* Matplotlib
* Joblib

---

# ▶️ Como Executar

## 1. Clonar o repositório

```bash
git clone https://github.com/eltonjsilva05-spec/deteccao-fraudes-cartao.git
```

## 2. Entrar na pasta

```bash
cd deteccao-fraudes-cartao
```

## 3. Criar ambiente virtual

```bash
python -m venv .venv
```

## 4. Ativar o ambiente

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

## 5. Instalar as dependências

```bash
pip install -r requirements.txt
```

## 6. Executar o projeto

```bash
python src/main.py
```

O sistema realizará automaticamente:

1. Download do dataset caso necessário;
2. Carregamento dos dados;
3. Pré-processamento;
4. Divisão entre treino e teste;
5. Balanceamento do treinamento com SMOTE;
6. Treinamento dos modelos;
7. Avaliação;
8. Comparação dos modelos;
9. Seleção do melhor modelo;
10. Geração das visualizações;
11. Análise de thresholds;
12. Geração da explicabilidade SHAP;
13. Salvamento dos modelos e resultados.

---

# 📌 Principais Resultados

O modelo XGBoost apresentou:

```text
Precision       85,21%
Recall          81,76%
F1-Score        83,45%
ROC-AUC         97,57%
Average Precision 84,05%
```

Após a análise de threshold:

```text
Threshold       0.77
Precision       89,47%
Recall          80,41%
F1-Score        84,70%
```

Esses resultados demonstram que a alteração do threshold pode melhorar o equilíbrio entre detecção de fraudes e redução de falsos positivos.

---

# 💡 Aprendizados

Durante o desenvolvimento foram trabalhados conceitos importantes de Machine Learning aplicado a problemas financeiros:

* Classificação binária;
* Dados altamente desbalanceados;
* SMOTE;
* Data leakage;
* Feature scaling;
* Comparação de modelos;
* Precision e Recall;
* F1-Score;
* ROC-AUC;
* Average Precision;
* Threshold optimization;
* Feature importance;
* Explainable AI;
* SHAP;
* Persistência de modelos com Joblib;
* Organização de projetos de Machine Learning em Python.

---

# 👨‍💻 Autor

**Elton Silva**

Projeto desenvolvido como parte do portfólio profissional, com foco em **Python, Machine Learning, Ciência de Dados e análise de problemas reais de negócio**.

GitHub:

```text
https://github.com/eltonjsilva05-spec
```
