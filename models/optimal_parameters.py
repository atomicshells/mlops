from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings

# Suppressing unnecessary warnings
warnings.filterwarnings("ignore")

def optimize_model(models, params_dict, X_train, y_train, X_test, y_test):
    """
    Optimizes models using GridSearchCV based on a defined parameter grid and returns the best parameters and R-squared score.

    Args:
        models (dict): A dictionary containing the models to tune.
        params_dict (dict): A dictionary containing the parameter grid for each model.
        X_train (DataFrame): Training features.
        y_train (Series): Training target variable.
        X_test (DataFrame): Testing features.
        y_test (Series): Testing target variable.

    Returns:
        dict: A dictionary containing the best parameters and test R2 scores for each model.
    """
    best_params_dict = {}
    for name, model in models.items():
        grid = GridSearchCV(estimator=model, param_grid=params_dict[name], scoring='r2', n_jobs=-1, cv=5, verbose=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        best_params_dict[name] = {'Best Parameters': grid.best_params_, 'Test R2 Score': test_r2}

    return best_params_dict

# Define models and parameter grids
params_dict = {
    # Parameter grids as defined earlier
}

models = {
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Support Vector Regression': SVR(kernel='linear'),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor()
}