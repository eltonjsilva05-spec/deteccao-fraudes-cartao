import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    MODELS_PATH
)


def dividir_dados(df):
    """
    Divide os dados em treino e teste.

    O StandardScaler é ajustado somente nos dados de treino,
    evitando data leakage.

    O SMOTE é aplicado somente ao conjunto de treino.
    """

    X = df.drop(
        TARGET_COLUMN,
        axis=1
    )

    y = df[TARGET_COLUMN]

    # ---------------------------------------------------------
    # 1. Divisão treino / teste
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------
    # 2. Normalização
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Amount_scaled"] = scaler.fit_transform(
        X_train[["Amount"]]
    )

    X_test["Amount_scaled"] = scaler.transform(
        X_test[["Amount"]]
    )

    # ---------------------------------------------------------
    # 3. Remover Amount original
    # ---------------------------------------------------------

    X_train = X_train.drop(
        "Amount",
        axis=1
    )

    X_test = X_test.drop(
        "Amount",
        axis=1
    )

    # ---------------------------------------------------------
    # 4. SMOTE somente no treinamento
    # ---------------------------------------------------------

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nDistribuição original do treino:")
    print(
        y_train.value_counts()
    )

    print("\nDistribuição após SMOTE:")
    print(
        y_train_smote.value_counts()
    )

    # ---------------------------------------------------------
    # 5. Salvar scaler
    # ---------------------------------------------------------

    salvar_scaler(
        scaler
    )

    return (
        X_train,
        X_train_smote,
        X_test,
        y_train,
        y_train_smote,
        y_test
    )


def treinar_logistic(X_train, y_train):
    """
    Treina Logistic Regression.

    Utiliza class_weight='balanced' para lidar com o
    desbalanceamento sem precisar utilizar SMOTE.

    O solver liblinear é utilizado para melhorar a
    convergência do modelo neste conjunto de dados.
    """

    modelo = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def treinar_random_forest(X_train, y_train):
    """
    Treina Random Forest utilizando o conjunto balanceado
    pelo SMOTE.
    """

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def treinar_xgboost(X_train, y_train):
    """
    Treina XGBoost utilizando o conjunto balanceado pelo SMOTE.

    O scale_pos_weight não é utilizado porque o conjunto
    de treinamento já foi balanceado.
    """

    modelo = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def salvar_scaler(scaler):
    """
    Salva o StandardScaler utilizado no treinamento.
    """

    os.makedirs(
        MODELS_PATH,
        exist_ok=True
    )

    caminho = os.path.join(
        MODELS_PATH,
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        caminho
    )

    print(
        f"Scaler salvo em: {caminho}"
    )


def salvar_modelo(modelo, nome_modelo):
    """
    Salva um modelo treinado.
    """

    os.makedirs(
        MODELS_PATH,
        exist_ok=True
    )

    caminho = os.path.join(
        MODELS_PATH,
        f"{nome_modelo}.pkl"
    )

    joblib.dump(
        modelo,
        caminho
    )

    print(
        f"Modelo salvo em: {caminho}"
    )