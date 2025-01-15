from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np

def train_models(X, y):
    """
    Trains multiple regression models on a given dataset and returns trained models along with their training and testing sets.

    Args:
        X (pd.DataFrame): The input features of the dataset.
        y (pd.Series): The target variable of the dataset.

    Returns:
        dict: A dictionary containing the trained models and their corresponding training and testing data.
              Each entry is keyed by model name with a tuple of (model, X_train, X_test, y_train, y_test).

    Note:
        The random_state is fixed to ensure reproducibility. Adjust the test_size and random_state as needed for different dataset sizes or reproducibility requirements.
    """
    random_state = 888
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

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = (model, X_train, X_test, y_train, y_test)

    return trained_models