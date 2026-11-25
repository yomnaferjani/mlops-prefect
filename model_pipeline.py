"""
model_pipeline.py
Fonctions modulaires du pipeline ML (prédiction du churn bancaire).
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """
    Charge le fichier CSV et le prétraite.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine aléatoire (reproductibilité).

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du fichier.")

    # Encodage de la colonne Gender (Female/Male -> 0/1)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Séparation variables explicatives / cible
    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un modèle RandomForest.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évalue le modèle et affiche les métriques.

    Returns:
        dict: accuracy et matrice de confusion.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :")
    print(matrix)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    return {"accuracy": accuracy, "confusion_matrix": matrix}


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. Lancez d'abord l'entraînement."
        )
    return joblib.load(path)
