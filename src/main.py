import pandas as pd
from src.features.build_features import build_features
from models.train_model import train_models
from models.predict import make_predictions
from models.evaluate_model import evaluate_model
from models.interpretability import plot_feature_importance
from models.optimal_parameters import optimize_model
from src.data.load_data import load_data

def main():
    """
    Main function to execute machine learning pipeline tasks: load data, build features, train models,
    make predictions, evaluate models, and optimize hyperparameters for the best performing model.
    """
    # Load data
    data = load_data('data/data.csv')

    # Build features
    X, y = build_features(data)

    # Train models and store results
    trained_models = train_models(X, y)
    performance_records = []

    # Evaluate and display results
    for name, (model, X_train, X_test, y_train, y_test) in trained_models.items():
        print(f"Evaluating {name}")

        # Predictions
        y_pred = make_predictions(model, X_test)

        # Evaluate
        r2 = evaluate_model(model, X_test, y_test)
        print(f"Test R^2 for {name}: {r2}")

        # Record performance for later use
        performance_records.append((name, model, r2, X_train, X_test, y_train, y_test))

        # Plot feature importance for models that support it
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, X_train.columns)

    # Determine best model based on R^2 score
    best_model_name, best_model, best_r2, best_X_train, best_X_test, best_y_train, best_y_test = max(performance_records, key=lambda x: x[2])

    # Define parameter grid for the best model, this needs to be predefined or dynamically created
    param_grids = {
        'AdaBoost': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1, 0.3]
        },
        # Add parameter grids for other models similarly
    }
    
    # Ensure the best model has a parameter grid defined
    if best_model_name in param_grids:
        best_params, best_params_r2 = optimize_model(best_model, param_grids[best_model_name], best_X_train, best_y_train, best_X_test, best_y_test)
        print(f"Best parameters for {best_model_name}: {best_params}")
        print(f"Best model test R^2 after optimization: {best_params_r2}")
    else:
        print(f"No parameter grid defined for {best_model_name}, skipping optimization.")

if __name__ == '__main__':
    main()
