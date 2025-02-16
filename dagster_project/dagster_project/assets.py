
import requests
from dagster import asset
import pandas as pd
from src.data import load_data
from src.models import train_models as train_model_fn
# from src.utils.save_utils import save_model

@asset
def raw_data():
    return load_data("data/data.csv")

@asset
def trained_model(raw_data):
    x_data = raw_data.drop("target", axis=1)    
    y_data = raw_data["target"] 
    model = train_model_fn(x_data, y_data) 
    return model

# @asset
# def saved_model(trained_model):
#     save_model(trained_model, "model.pkl")
#     return "model.pkl"
