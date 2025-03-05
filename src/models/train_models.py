from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets, trains a RandomForestRegressor, and returns the trained model and test set.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        RandomForestRegressor: Trained model.
        pd.DataFrame: X_test (features for test set).
        pd.Series: y_test (labels for test set).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=50, random_state=random_state)
    model.fit(X_train, y_train)

    return model, X_test, y_test