import pandas as pd

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