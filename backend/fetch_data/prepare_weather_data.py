import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

input_file = "backend/dataset/vietnam_weather_hourly.csv"
processed_folder = "backend/dataset/processed"
os.makedirs(processed_folder, exist_ok=True)
output_train = os.path.join(processed_folder, "train_data.csv")
output_test = os.path.join(processed_folder, "test_data.csv")

if os.path.exists(output_train):
    os.remove(output_train)
if os.path.exists(output_test):
    os.remove(output_test)

dtype_mapping = {
    "Latitude": "float32",
    "Longitude": "float32",
    "WS10M": "float32",
    "QV2M": "float32",
    "T2M": "float32",
    "PS": "float32",
    "PRECTOTCORR": "float32"
}

chunk_size = 25000
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=chunk_size, dtype=dtype_mapping, parse_dates=["Datetime"], low_memory=True):
    chunk.dropna(inplace=True)
    chunk["hour"] = chunk["Datetime"].dt.hour
    chunk["day"] = chunk["Datetime"].dt.day
    chunk["month"] = chunk["Datetime"].dt.month
    chunk["season"] = (chunk["month"] % 12 + 3) // 3 
    
    chunk = chunk.sort_values(by="Datetime")
    
    train_size = int(len(chunk) * 0.8)
    train_chunk = chunk.iloc[:train_size]
    test_chunk = chunk.iloc[train_size:]
    
    train_chunk.to_csv(output_train, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
    test_chunk.to_csv(output_test, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
    
    first_chunk = False 

print("✅ Dữ liệu đã được xử lý từng phần và lưu vào tập train/test.")
