import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(data):
    """
    Handles missing values in a DataFrame by filling them with the mean of their respective columns.

    Args:
        data (pd.DataFrame): The DataFrame with missing values to handle.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
    """
    missing_values_before = data.isnull().sum()
    print(f"Number of missing values per feature prior to imputation:\n\n{missing_values_before}\n")

    data_filled = data.fillna(data.mean())

    missing_values_after = data_filled.isnull().sum()
    print(f"Number of missing values per feature after imputation:\n\n{missing_values_after}")
    
    return data_filled

def remove_duplicates(data):
    """
    Removes duplicate rows from a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame from which to remove duplicates.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    return data.drop_duplicates()

def scale_features(data):
    """
    Scales all numerical features in a DataFrame using StandardScaler.

    Args:
        data (pd.DataFrame): The DataFrame containing the features to scale.

    Returns:
        pd.DataFrame: The DataFrame with all numerical features scaled.
    """
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=[np.number]).columns  # Automatically select numerical columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data