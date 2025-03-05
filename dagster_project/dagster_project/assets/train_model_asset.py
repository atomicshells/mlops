from dagster import asset
from src.models.train_models import train_model

@asset
def train_model_asset(build_features):
    X_train_scaled = build_features["X_train_scaled"]
    y_train = build_features["y_train"]

    model = train_model(X_train_scaled, y_train)
    return model