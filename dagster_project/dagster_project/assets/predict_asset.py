from dagster import asset
from src.models.predict import predict
from src.models.evaluate_model import evaluate_model

@asset
def predict_and_evaluate(train_model_asset, build_features):
    X_test_scaled = build_features["X_test_scaled"]
    y_test = build_features["y_test"]

    y_pred = predict(train_model_asset, X_test_scaled)
    evaluate_model(y_test, y_pred)