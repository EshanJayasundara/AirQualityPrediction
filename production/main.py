import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.utils import *
from typing import List, Dict
import numpy as np
import pandas as pd

with open("model_name.txt", "r") as f:
    model_name = f.readline().strip()
    f.close()

mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow server URI
client = MlflowClient()

# Get latest production version
latest_versions = client.get_latest_versions(model_name, stages=["Production"])

if not latest_versions:
    raise ValueError(f"No model found in 'Production' stage for: {model_name}")

model_version = latest_versions[0].version  # Get the latest production version

# Load the model using the version
MODEL_URI = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()

# Define request body
class PredictionRequest(BaseModel):
    features: List[List[float]]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        X_preprocessed = preprocess_production(request.features, pickle_loc="caches.pkl")
        predictions = model.predict(X_preprocessed)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health_check():
    return {"message": "MLflow API is running"}
