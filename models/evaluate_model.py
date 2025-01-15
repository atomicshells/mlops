from sklearn.metrics import r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluates each model provided and returns their R2 scores.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (DataFrame): Test features.
        y_test (Series): Test target variable.
    
    Returns:
        DataFrame: Model performances.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        results.append([name, test_r2])
    return pd.DataFrame(results, columns=['Model', 'Test R2'])