from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluates the regression model using Mean Squared Error and R^2 Score.

    Args:
        y_true (pd.Series): True target values.
        y_pred (np.array): Predicted values.

    Returns:
        None
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
