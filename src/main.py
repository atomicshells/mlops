import time
from src.data.load_data import load_data
from src.data.preprocess_data import handle_missing_values, remove_duplicates
from src.features.build_features import scale_features
from src.models.train_models import train_model
from src.models.predict import predict
from src.models.evaluate_model import evaluate_model

print("Pipeline starting...")

def main():
    """
    Main function to orchestrate the ML pipeline.
    """

    start = time.time()

    # Step 1: Load data
    print("Loading data...")
    df = load_data('data/data.csv')
    print(f"Data loading took {time.time() - start:.2f} seconds")

    # Step 2: Preprocess data
    start = time.time()
    print("Preprocessing data...")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    print(f"Preprocessing took {time.time() - start:.2f} seconds")

    # Step 3: Feature engineering
    start = time.time()
    target_column = 'Global Innovation Score'
    exclude_columns = [target_column]
    X, scaler = scale_features(df, exclude_columns=exclude_columns)
    y = df[target_column]
    print(f"Feature engineering took {time.time() - start:.2f} seconds")

    # Step 4: Train model (with train-test split)
    start = time.time()
    print("Training model...")
    model, X_test, y_test = train_model(X, y)
    print(f"Model training took {time.time() - start:.2f} seconds")

    # Step 5: Make predictions and evaluate
    start = time.time()
    print("Making predictions and evaluating...")
    y_pred = predict(model, X_test)
    evaluate_model(y_test, y_pred)
    print(f"Prediction and evaluation took {time.time() - start:.2f} seconds")

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()