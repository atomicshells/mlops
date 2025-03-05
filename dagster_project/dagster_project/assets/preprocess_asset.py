from dagster import asset
from src.data.preprocess_data import handle_missing_values, remove_duplicates, split_train_test

@asset
def preprocess_data(load_data):
    df = handle_missing_values(load_data)
    df = remove_duplicates(df)

    target_column = 'Global Innovation Score'
    y = df[target_column]
    X = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }