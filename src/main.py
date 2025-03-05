import time
from src.data.load_data import load_data
from src.data.preprocess_data import handle_missing_values, remove_duplicates, split_train_test
from src.features.build_features import scale_features
from src.models.train_models import train_model
from src.models.predict import predict
from src.models.evaluate_model import evaluate_model

def main():
    print("Pipeline starting...")

    start = time.time()

    # Step 1: Load data
    print("Loading data...")
    df = load_data('data/data.csv')
    print(f"Data loading took {time.time() - start:.2f} seconds")

    # Step 2: Preprocess data (cleaning + splitting)
    start = time.time()
    print("Preprocessing data...")
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    target_column = 'Global Innovation Score'
    y = df[target_column]
    X = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print(f"Preprocessing took {time.time() - start:.2f} seconds")

    # Step 3: Feature engineering (scaling)
    start = time.time()
    print("Feature engineering...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"Feature engineering took {time.time() - start:.2f} seconds")

    # Step 4: Train model
    start = time.time()
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    print(f"Model training took {time.time() - start:.2f} seconds")

    # Step 5: Predict & evaluate
    start = time.time()
    print("Making predictions and evaluating...")
    y_pred = predict(model, X_test_scaled)
    evaluate_model(y_test, y_pred)
    print(f"Prediction and evaluation took {time.time() - start:.2f} seconds")

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()