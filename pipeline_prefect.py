"""Pipeline Prefect - Atelier 3 MLOps."""

import argparse
import os
import subprocess
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# Étape 4 : on ne cible QUE nos 3 fichiers
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
MODEL_PATH = "classifier.joblib"  # même valeur par défaut que save_model/load_model

# Étape 6 : à remplir plus tard avec l'URL de ton repo GitHub
REPO_URL = "https://github.com/TON_USER/mlops-prefect.git"
REPO_DIR = "mlops-prefect-clone"


def run_cmd(cmd, check=True):
    """Exécute une commande système et écrit sa sortie dans les logs Prefect."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    return result.returncode


# ------------------------------------------------------------------ TASKS
@task(name="Récupérer le projet")
def clone_repo():
    if os.path.isdir(REPO_DIR):
        run_cmd(["git", "-C", REPO_DIR, "pull"])
    else:
        run_cmd(["git", "clone", REPO_URL, REPO_DIR])


@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="Qualité du code")
def check_quality():
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *FILES],
        check=False,
    )


@task(name="Sécurité du code")
def check_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", *FILES], check=False)


@task(name="Tests unitaires")
def run_unit_tests():
    run_cmd([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


@task(name="Préparation des données")
def prepare_data_task():
    return prepare_data()  # lit Churn_Modelling.csv (valeur par défaut)


@task(name="Entraînement du modèle")
def train_model_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="Sauvegarder le modèle")
def save_model_task(model):
    save_model(model, MODEL_PATH)


@task(name="Charger le modèle")
def load_model_task():
    return load_model(MODEL_PATH)


@task(name="Évaluer le modèle")
def evaluate_model_task(model, x_test, y_test):
    metrics = evaluate_model(model, x_test, y_test)
    get_run_logger().info("Résultats : %s", metrics)
    return metrics


# ------------------------------------------------------------------ FLOWS
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()  # <-- ADAPTE
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    model = load_model_task()
    _, x_test, _, y_test = prepare_data_task()  # <-- ADAPTE
    evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    # clone_repo()  # <-- à décommenter à l'étape 6 (repo Git)
    install_dependencies()
    code_flow()
    x_train, x_test, y_train, y_test = prepare_data_task()  # <-- ADAPTE
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


# -------------------------------------------------------------------- CLI
FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,  # alias utilisé dans l'énoncé
    "evaluate": evaluate_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
