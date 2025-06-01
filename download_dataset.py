import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Define the path where the dataset will be downloaded
download_path = 'Data/'
os.makedirs(download_path, exist_ok=True)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

try:
    print("Starting download...")
    api.dataset_download_files('parvmodi/automotive-vehicles-engine-health-dataset', path=download_path, unzip=True)
    print("Dataset downloaded and extracted successfully.")
except Exception as e:
    print(f"Error: {e}")