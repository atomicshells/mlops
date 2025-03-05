import pandas as pd

def load_data(filepath='data/data.csv'):
    """
    Loads data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(filepath)
    return df
