import time

import pandas as pd
import requests as rq

from diabetes_prediction_api import ROOT_DIR
from diabetes_prediction_api.utils import draw_roc

API_URI = "http://127.0.0.1:5000"


def print_response(response: rq.Response) -> None:
    print("request.url: ", response.url)
    print("request.status_code: ", response.status_code)
    print("request.headers: ", response.headers)
    print("request.text: ", response.text)
    print("request.request.body: ", response.request.body)
    print("request.request.headers: ", response.request.headers)
    print("---------------------------------------------------------------")


def run_test():
    payload = {
        "completed_job_ID": str(hash(time.time())),
        "new_model_file_name": "diabetes_clf.joblib",
    }
    rq.put(f"{API_URI}/model", json=payload)

    X_test = pd.read_csv(f"{ROOT_DIR}/train/data//X_test.csv")
    y_test = pd.read_csv(f"{ROOT_DIR}/train/data/y_test.csv")

    predictions = []
    for row in X_test.iterrows():
        print(row)
        response = rq.post(f"{API_URI}/patient-records", json=row[1].to_dict())
        print_response(response)
        predictions.append(response.json()["probability_of_diabetes"])

    draw_roc(y_test, predictions, "Online tests")
