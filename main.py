"""
main.py
Point d'entrée : exécute les étapes du pipeline via des arguments CLI.
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "load", "all"],
        required=True,
        help="Étape à exécuter",
    )
    parser.add_argument("--data", default="Churn_Modelling.csv", help="Fichier CSV")
    parser.add_argument("--model", default="classifier.joblib", help="Fichier modèle")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print("Données préparées.")
        print(f"x_train : {x_train.shape} | x_test : {x_test.shape}")

    elif args.step == "train":
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    elif args.step == "evaluate":
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)
    elif args.step == "load":
        model = load_model(args.model)
        print("Modèle chargé :", type(model).__name__)
        print("Nombre d'arbres :", model.n_estimators)

    elif args.step == "all":
        model = train_model(x_train, y_train)
        evaluate_model(model, x_test, y_test)
        save_model(model, args.model)
        loaded = load_model(args.model)
        print("Modèle rechargé :", type(loaded).__name__)


if __name__ == "__main__":
    main()
