from dagster import asset
from src.features.build_features import scale_features

@asset
def build_features(preprocess_data):
    X_train, X_test = preprocess_data["X_train"], preprocess_data["X_test"]
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": preprocess_data["y_train"],
        "y_test": preprocess_data["y_test"],
        "scaler": scaler
    }
