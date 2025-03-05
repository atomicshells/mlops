from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(X_train, X_test):
    """
    Fits a StandardScaler on the training set and applies it to both train and test sets.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.

    Returns:
        pd.DataFrame: Scaled X_train.
        pd.DataFrame: Scaled X_test.
        StandardScaler: Fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
           pd.DataFrame(X_test_scaled, columns=X_test.columns), \
           scaler
