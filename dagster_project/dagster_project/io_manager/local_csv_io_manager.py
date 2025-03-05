import os
import pandas as pd
from dagster import IOManager, InputContext, OutputContext

DATA_FOLDER = "data/asset_outputs"

class LocalCSVIOManager(IOManager):
    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        os.makedirs(DATA_FOLDER, exist_ok=True)
        file_path = os.path.join(DATA_FOLDER, f"{context.asset_key.to_user_string()}.csv")
        obj.to_csv(file_path, index=False)
        context.log.info(f"Saved asset '{context.asset_key.to_user_string()}' to {file_path}")

    def load_input(self, context: InputContext) -> pd.DataFrame:
        file_path = os.path.join(DATA_FOLDER, f"{context.asset_key.to_user_string()}.csv")
        context.log.info(f"Loading asset '{context.asset_key.to_user_string()}' from {file_path}")
        return pd.read_csv(file_path)

local_csv_io_manager = LocalCSVIOManager()