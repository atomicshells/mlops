from dagster import Definitions
from dagster_project.assets.load_data_asset import load_data
from dagster_project.assets.preprocess_asset import preprocess_data
from dagster_project.assets.features_asset import build_features
from dagster_project.assets.train_model_asset import train_model_asset
from dagster_project.assets.predict_asset import predict_and_evaluate
from dagster_project.io_manager.local_csv_io_manager import local_csv_io_manager

defs = Definitions(
    assets=[load_data, preprocess_data, build_features, train_model_asset, predict_and_evaluate],
    resources={
        "io_manager": local_csv_io_manager
    }
)