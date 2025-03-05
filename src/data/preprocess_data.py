from sklearn.model_selection import train_test_split

def handle_missing_values(df):
    """
    Fills missing values with the column mean.

    Args:
        df (pd.DataFrame): Input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    return df.fillna(df.mean())

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows.
    """
    return df.drop_duplicates()

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target column.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test (tuple of DataFrames/Series)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
