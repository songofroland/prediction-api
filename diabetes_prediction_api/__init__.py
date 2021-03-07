from pathlib import Path

from flask import Flask, jsonify
from redis import Redis

app = Flask(__name__)
redis = Redis(decode_responses=True)
ROOT_DIR = Path(__file__).parents[1]
config = {"model_file_name": f"{ROOT_DIR}/models/diabetes_clf.joblib"}


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404
