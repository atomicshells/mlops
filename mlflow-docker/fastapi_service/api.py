# type: ignore

from fastapi import FastAPI
import mlflow
import pandas as pd

app = FastAPI()

# Load latest production model from MLflow Model Registry
model = mlflow.sklearn.load_model("models:/GlobalInnovationModel/Production")

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
