from sklearn.model_selection import GridSearchCV

def optimize_model(model, param_grid, X_train, y_train):
    """
    Optimizes a model using GridSearchCV to find the best parameters.

    Args:
        model (model object): The model to be optimized.
        param_grid (dict): A dictionary specifying the parameters to be tested.
        X_train (pd.DataFrame): The training set features.
        y_train (pd.Series): The target values for the training set.

    Returns:
        tuple: A tuple containing the best model and its parameters.

    Description:
        This function uses cross-validation to determine the best parameters for the model,
        given the parameter grid.
    """
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', n_jobs=-1, cv=5, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_