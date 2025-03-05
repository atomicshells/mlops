from sklearn.metrics import accuracy_score

def evaluate_model(y_true, y_pred):
    """
    Evaluates the model using accuracy.

    Args:
        y_true (pd.Series): True target values.
        y_pred (np.array): Predicted values.

    Returns:
        None
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")