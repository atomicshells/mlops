import pandas as pd
from dagster import asset

@asset
def load_data():
    return pd.read_csv('data/data.csv')
