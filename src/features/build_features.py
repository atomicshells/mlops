from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(df, exclude_columns=None):
    """
    Scales numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): Input DataFrame with features.
        exclude_columns (list): List of columns to exclude from scaling (e.g., target column).

    Returns:
        pd.DataFrame: Scaled features.
        StandardScaler: Fitted scaler object.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    features = df.drop(columns=exclude_columns)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    for col in exclude_columns:
        scaled_df[col] = df[col].values

    return scaled_df, scaler