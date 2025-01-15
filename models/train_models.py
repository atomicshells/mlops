from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def train_models(X, y, random_state=888):
    """
    Trains multiple regression models and returns them with their training data split.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.
        random_state (int): Seed for the random number generator.
    
    Returns:
        dict: Dictionary containing trained models and their data splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    models = {
        'Ridge Regression': Ridge(random_state=random_state),
        'Lasso Regression': Lasso(random_state=random_state),
        'Support Vector Regression': SVR(kernel='linear'),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'AdaBoost': AdaBoostRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models, X_train, X_test, y_train, y_test