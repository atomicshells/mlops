def build_features(data, target_column='Global Innovation Score'):
    """
    Splits the dataset into features and target.

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): The name of the target column. Defaults to 'Global Innovation Score'.

    Returns:
        pd.DataFrame: Features (X).
        pd.Series: Target (y).
    """
    X = data.drop(target_column, axis=1)  # Remove the target column to create the features set
    y = data[target_column]  # The target column

    return X, y