# data_utils.py

import os
import zipfile
import pathlib
import pandas as pd

def unzip_and_load_data(data_dir='./data/', zip_file_path='optiver-realized-volatility-prediction.zip'):
    """
    Handles unzipping the dataset and loading the initial train.csv file.
    """
    print("Attempting to find and unzip the local data file...")
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    try:
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Successfully extracted '{zip_file_path}' to '{data_dir}'")

    except FileNotFoundError:
        print("---")
        print(f"FATAL ERROR: The file '{zip_file_path}' was not found.")
        print("Please upload the Kaggle dataset zip file and re-run.")
        print("---")
        return None, None, None

    train_file = os.path.join(data_dir, 'train.csv')
    if os.path.exists(train_file):
        try:
            train = pd.read_csv(train_file)
            stock_ids = train['stock_id'].unique()
            time_ids = train['time_id'].unique()
            print(f"Successfully loaded {train_file}.")
            print(f"Number of unique stock_ids: {len(stock_ids)}")
            print(f"Number of unique time_ids: {len(time_ids)}")
            return train, stock_ids, time_ids
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            return None, None, None
    else:
        print(f"Could not proceed to load data because '{train_file}' does not exist.")
        return None, None, None

def load_book_data(stock_id, data_dir='./data/'):
    path = os.path.join(data_dir, f'book_train.parquet/stock_id={stock_id}')
    return pd.read_parquet(path)

def load_trade_data(stock_id, data_dir='./data/'):
    path = os.path.join(data_dir, f'trade_train.parquet/stock_id={stock_id}')
    return pd.read_parquet(path)