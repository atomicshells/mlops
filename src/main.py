from src.data.load_data import load_data
from src.data.preprocess_data import handle_missing_values, remove_duplicates
from src.features.build_features import scale_features
from src.models.train_models import train_model
from src.models.predict import predict
from src.models.evaluate_model import evaluate_model

def main():
    """
    Main function to orchestrate the ML pipeline.
    """
    df = load_data('data/data.csv')

    # Preprocess data
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    # Set target column
    target_column = 'Global Innovation Score'  
    exclude_columns = [target_column]

    # Feature engineering
    X, scaler = scale_features(df, exclude_columns=exclude_columns)
    y = df[target_column]

    # Train model (with train-test split)
    model, X_test, y_test = train_model(X, y)

    # Predict on test set and evaluate
    y_pred = predict(model, X_test)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()