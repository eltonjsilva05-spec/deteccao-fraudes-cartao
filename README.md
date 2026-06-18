# 🚨 Detecção de Fraudes em Cartões de Crédito

Projeto desenvolvido como parte dos estudos de Ciência de Dados e Machine Learning, com foco na identificação de transações fraudulentas utilizando diferentes algoritmos de classificação.

---

# 📖 Sobre o Projeto

Fraudes em cartões de crédito representam um grande desafio para instituições financeiras devido ao alto volume de transações e ao forte desbalanceamento entre operações legítimas e fraudulentas.

Neste projeto foram aplicadas técnicas de pré-processamento, balanceamento de classes e algoritmos de Machine Learning para detectar transações suspeitas.

---

# 🎯 Objetivos

- Analisar um conjunto de dados real de transações financeiras.
- Tratar o problema de desbalanceamento de classes.
- Comparar diferentes algoritmos de classificação.
- Avaliar modelos utilizando métricas adequadas para detecção de fraudes.
- Selecionar automaticamente o melhor modelo.

---

# 📊 Dataset

O projeto utiliza o dataset público:

Credit Card Fraud Detection Dataset

Características:

- 284.807 transações
- 492 fraudes
- Dataset altamente desbalanceado
- Variáveis anonimizadas através de PCA
- Variável alvo: `Class`

Onde:

- 0 = Transação Normal
- 1 = Fraude

Fonte:

https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

---

# 🔄 Fluxo do Projeto

```text
Dataset
   ↓
Carregamento Automático
   ↓
Pré-processamento
   ↓
Transformação Logarítmica
   ↓
Padronização (StandardScaler)
   ↓
Train/Test Split
   ↓
SMOTE
   ↓
Treinamento dos Modelos
   ↓
Avaliação
   ↓
Comparação dos Resultados
   ↓
Seleção do Melhor Modelo
```

---

# 🧠 Técnicas Utilizadas

## Pré-processamento

### Transformação Logarítmica

Aplicada na variável:

```python
Amount
```

Objetivo:

- Reduzir assimetria dos dados
- Melhorar desempenho dos algoritmos

### Padronização

Utilizando:

```python
StandardScaler()
```

Objetivo:

- Colocar variáveis na mesma escala
- Melhorar convergência dos modelos

---

# ⚖️ Balanceamento das Classes

O dataset possui pouquíssimas fraudes.

Para corrigir esse problema foi utilizado:

```python
SMOTE
```

(Synthetic Minority Oversampling Technique)

Objetivo:

- Gerar exemplos sintéticos da classe minoritária.
- Melhorar a capacidade de detecção de fraudes.

---

# 🤖 Modelos Utilizados

## Logistic Regression

Modelo linear utilizado como baseline.

Características:

- Simples
- Rápido
- Fácil interpretação

---

## Random Forest

Modelo baseado em múltiplas árvores de decisão.

Características:

- Reduz overfitting
- Boa performance em dados tabulares

---

## XGBoost

Modelo de Gradient Boosting.

Características:

- Alta performance
- Excelente para datasets tabulares
- Amplamente utilizado em projetos reais

---

# 📈 Métricas Avaliadas

Os modelos são avaliados utilizando:

## Recall

Métrica mais importante para fraude.

Representa:

> Quantas fraudes reais foram identificadas.

---

## Precision

Representa:

> Quantas previsões de fraude realmente eram fraudes.

---

## F1-Score

Média harmônica entre:

- Precision
- Recall

---

## ROC AUC

Avalia a capacidade de separação entre as classes.

---

# 📂 Estrutura do Projeto

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
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🗂 Descrição dos Arquivos

## config.py

Armazena as configurações gerais do projeto.

---

## data_loader.py

Responsável pelo carregamento dos dados e download automático do dataset.

---

## preprocessing.py

Executa:

- Transformação logarítmica
- Padronização dos dados

---

## train.py

Responsável por:

- Divisão treino/teste
- Aplicação do SMOTE
- Treinamento dos modelos
- Salvamento dos modelos

---

## evaluate.py

Responsável por:

- Classification Report
- ROC AUC
- Curva ROC
- Matriz de Confusão
- Comparação entre modelos

---

## explainability.py

Implementa interpretabilidade utilizando:

```python
SHAP
```

---

## main.py

Arquivo principal que executa todo o pipeline.

---

# 📁 Arquivos Gerados

Após a execução serão gerados:

## Models

```text
models/
├── logistic.pkl
├── random_forest.pkl
├── xgboost.pkl
└── best_model.pkl
```

---

## Results

```text
results/
├── metricas.csv
├── XGBClassifier_roc.png
└── XGBClassifier_confusion_matrix.png
```

---

# 🚀 Como Executar

## Clonar o repositório

```bash
git clone URL_DO_REPOSITORIO
```

---

## Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Executar

```bash
python src/main.py
```

---

# 📊 Comparação dos Modelos

Ao executar o projeto é gerado:

```text
results/metricas.csv
```

Exemplo:

| Modelo | Recall | Precision | F1 |
|----------|----------|----------|----------|
| Logistic Regression | 0.84 | 0.91 | 0.87 |
| Random Forest | 0.91 | 0.95 | 0.93 |
| XGBoost | 0.95 | 0.97 | 0.96 |

O melhor modelo é selecionado automaticamente e salvo como:

```text
models/best_model.pkl
```

---

# 🔍 Interpretabilidade

O projeto utiliza:

```python
SHAP
```

para identificar quais variáveis possuem maior impacto nas previsões realizadas pelo modelo.

---

# 📌 Melhorias Futuras

- GridSearchCV
- Feature Importance completa
- Dashboard com Streamlit
- API com FastAPI
- Docker
- Deploy em Cloud
- Monitoramento de modelos

---

# 👨‍💻 Autor

Projeto desenvolvido durante os estudos de:

- Python
- Ciência de Dados
- Machine Learning
- Detecção de Fraudes

Bootcamp DIO - Data Science & Machine Learning.
