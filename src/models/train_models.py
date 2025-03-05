from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.

    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    return model
