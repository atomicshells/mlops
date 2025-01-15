import os
import pandas as pd

def load_data(filename="data.csv"):
    """
    Loads a CSV file from the data folder.
    
    This function dynamically constructs the path to the 'data' directory
    to ensure that the CSV file is loaded correctly regardless of where the script is run
    from within the project structure.
    
    Args:
        filename (str): Name of the CSV file to load. Defaults to "data.csv".
    
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the specified file does not exist in the 'data' directory.
    """
    # Get the absolute path to the project folder
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", filename)
    
    # Check if the file exists at the path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the data folder.")
    
    return pd.read_csv(data_path)