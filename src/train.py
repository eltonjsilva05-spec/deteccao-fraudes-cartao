import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE, TEST_SIZE


def dividir_dados(df):
    """
    Divide os dados em treino e teste e aplica SMOTE.
    """

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nDistribuição após SMOTE:")
    print(y_train.value_counts())

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def treinar_logistic(X_train, y_train):
    """
    Treina Logistic Regression.
    """

   
    modelo = LogisticRegression(
    solver="saga",
    max_iter=5000,
    random_state=RANDOM_STATE
)

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def treinar_random_forest(X_train, y_train):
    """
    Treina Random Forest.
    """

    modelo = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
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
    Treina XGBoost.
    """

    modelo = XGBClassifier(
        scale_pos_weight=10,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def salvar_modelo(modelo, nome_modelo):
    """
    Salva o modelo treinado.
    """

    os.makedirs(
        "../models",
        exist_ok=True
    )

    caminho = f"../models/{nome_modelo}.pkl"

    joblib.dump(
        modelo,
        caminho
    )

    print(
        f"Modelo salvo em: {caminho}"
    )