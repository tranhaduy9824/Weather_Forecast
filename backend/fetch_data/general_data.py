import pandas as pd
import os
import glob
import gc

chunk_folder = "backend/dataset/weather_chunks"
output_csv = "backend/dataset/vietnam_weather_hourly.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

files = glob.glob(os.path.join(chunk_folder, "weather_*.csv"))

if not files:
    print("❌ Không có dữ liệu để gộp! Kiểm tra lại quá trình lấy dữ liệu.")
    exit()

dtype_mapping = {
    "Datetime": "str",
    "Latitude": "float32",
    "Longitude": "float32",
    "CLRSKY_SFC_SW_DWN": "float32",
    "WS10M": "float32",
    "QV2M": "float32",
    "T2M": "float32",
    "PS": "float32",
    "PRECTOTCORR": "float32"
}

chunk_size = 25000
first_file = True 

for file in files:
    try:
        print(f"📂 Đang xử lý file: {file}")
        df_chunks = pd.read_csv(file, chunksize=chunk_size, dtype=dtype_mapping, on_bad_lines='skip')

        for chunk in df_chunks:
            chunk["Datetime"] = pd.to_datetime(chunk["Datetime"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            chunk.dropna(subset=["Datetime"], inplace=True)  # Xóa hàng lỗi
            chunk = chunk.sort_values(by=["Datetime", "Latitude", "Longitude"])

            chunk.to_csv(output_csv, mode='w' if first_file else 'a', index=False, sep=",", float_format="%.2f", header=first_file)

            first_file = False

            del chunk
            gc.collect()

    except Exception as e:
        print(f"⚠️ Bỏ qua file {file} do lỗi: {e}")

print(f"✅ Dữ liệu đã được gộp và lưu vào {output_csv}")
