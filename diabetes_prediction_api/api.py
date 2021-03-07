import os
from hashlib import sha1

import numpy as np
from flask import request, abort

from . import config, redis, app, ROOT_DIR
from .utils import get_predictor


@app.route("/patient-records", methods=["POST"])
def create_entry():
    new_record = np.reshape(tuple(request.json.values()), (1, -1))
    prediction = get_predictor().predict_proba(new_record)[:, 0][0]
    patient_id = sha1(prediction).hexdigest()
    redis.set(patient_id, prediction)
    return {"patient_id": patient_id, "probability_of_diabetes": prediction}


@app.route("/patient-predictions/<patient_id>", methods=["GET"])
def get_entry(patient_id: str):
    resp = redis.get(patient_id)
    if resp is None:
        abort(404, description=f"Prediction for patient with id {patient_id} does not exist.")
    return {"patient_id": patient_id, "probability_of_diabetes": float(resp)}


@app.route("/model", methods=["PUT"])
def load_model():
    new_model_name = request.json["new_model_file_name"]
    if not os.path.isfile(f'{ROOT_DIR}/models/{new_model_name}'):
        abort(404, description=f"The file with name {new_model_name} does not exist.")
    config["model_file_name"] = new_model_name
    return {"result": "Model loaded successfully."}
