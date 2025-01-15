from sklearn.metrics import r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using the R-squared metric.

    Args:
        model (model object): A trained model object from scikit-learn or compatible libraries.
        X_test (pd.DataFrame): The test set features.
        y_test (pd.Series): The actual target values for the test set.

    Returns:
        float: The R-squared value indicating the model's performance.
    """
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)