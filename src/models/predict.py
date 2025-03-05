def predict(model, X):
    """
    Generates predictions using a trained model.

    Args:
        model (sklearn model): Trained model.
        X (pd.DataFrame): Feature matrix.

    Returns:
        np.array: Predictions.
    """
    return model.predict(X)