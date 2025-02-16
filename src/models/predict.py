def make_predictions(model, X):
    """
    Makes predictions using a trained model on the provided dataset.

    Args:
        model (model object): A trained model object from scikit-learn or compatible libraries.
        X (pd.DataFrame): The input features to make predictions on.

    Returns:
        np.array: An array of predictions made by the model.
    """
    return model.predict(X)