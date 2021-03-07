import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from diabetes_prediction_api import ROOT_DIR
from diabetes_prediction_api.utils import draw_roc

DATA_DIR = f"{ROOT_DIR}/train/data/"


def train(dev: bool = False) -> None:
    diabetes = pd.read_csv(f"{DATA_DIR}/diabetes.csv")
    X = diabetes.loc[:, diabetes.columns != "Outcome"]
    y = diabetes["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)

    mlp_config = {
        "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
    }
    if dev:
        mlp_config = {
            "hidden_layer_sizes": [(50,)],
            "activation": ["tanh"],
            "solver": ["adam"],
            "alpha": [0.05],
            "learning_rate": ["adaptive"],
        }
    clf = GridSearchCV(MLPClassifier(), mlp_config, n_jobs=1, cv=3)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_

    clf = MLPClassifier(**best_params)
    clf.fit(X_train, y_train)
    model_name = "diabetes_clf"
    dump(clf, f"{ROOT_DIR}/models/{model_name}.joblib")
    clf = load(f"{ROOT_DIR}/models/{model_name}.joblib")
    predictions = clf.predict_proba(X_test)[:, 0]

    draw_roc(y_test, predictions, "Offline tests")
