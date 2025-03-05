from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

def optimize_model(model, param_grid, X_train, y_train):
    """
    Optimizes a given model using GridSearchCV based on the parameter grid.

    Args:
        model: The model to be optimized.
        param_grid (dict): Parameter grid for the model.
        X_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        The model fitted with the best parameters.
    """
    grid = GridSearchCV(model, param_grid, scoring=make_scorer(r2_score), cv=5, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
