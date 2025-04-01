import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product

# Danh sách tọa độ
latitudes = np.arange(8, 25, 1)
longitudes = np.arange(102, 119, 1)

start_date = "20220101"
end_date = "20250331"  # Gộp toàn bộ khoảng thời gian

chunk_folder = "backend/dataset/weather_chunks"
os.makedirs(chunk_folder, exist_ok=True)

# Danh sách cột cố định
PARAMETERS = ["T2M", "QV2M", "PS", "WS10M", "PRECTOTCORR", "CLRSKY_SFC_SW_DWN"]
COLUMNS = ["Datetime", "Latitude", "Longitude"] + PARAMETERS

def fetch_weather_data(lat, lon):
    output_file = f"{chunk_folder}/weather_{lat}_{lon}.csv"

    if os.path.exists(output_file):
        print(f"✅ Bỏ qua {lat}, {lon}, đã có dữ liệu.")
        return

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    max_retries = 5  # Giảm số lần thử
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=15)  # Giảm timeout xuống 15s
            response.raise_for_status()

            data = response.json()
            if "properties" not in data or "parameter" not in data["properties"]:
                print(f"⚠️ API không trả dữ liệu hợp lệ cho {lat}, {lon}. Bỏ qua.")
                return

            parameters_data = data["properties"]["parameter"]
            print(f"📊 API trả về dữ liệu cho {lat}, {lon}: {len(parameters_data['T2M'])} thời điểm")

            # Ghi trực tiếp vào file
            with open(output_file, 'w') as f:
                f.write(",".join(COLUMNS) + "\n")  # Ghi header

                dates = sorted(parameters_data.get("T2M", {}).keys())  # YYYYMMDDHH
                for date_str in dates:
                    dt = pd.to_datetime(date_str, format="%Y%m%d%H").strftime("%Y-%m-%d %H:00:00")
                    row = [dt, str(lat), str(lon)]

                    for param in PARAMETERS:
                        value = parameters_data.get(param, {}).get(date_str, -999)
                        row.append(str(value))

                    f.write(",".join(row) + "\n")

            print(f"✅ Dữ liệu đã lưu thành công tại {output_file}")
            break

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            retry_count += 1
            print(f"⏳ Lỗi API tại {lat}, {lon}: {e}. Thử lại {retry_count}/{max_retries}...")
            if retry_count < max_retries:
                time.sleep(0.5)  # Giảm từ 3s xuống 0.5s
            else:
                print(f"❌ Bỏ qua {lat}, {lon} sau {max_retries} lần thử.")

lat_lon_pairs = list(product(latitudes, longitudes))

# Tăng max_workers lên 20 để tối đa hóa tốc độ
with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(lambda pair: fetch_weather_data(pair[0], pair[1]), lat_lon_pairs)

print("✅ Hoàn thành lấy dữ liệu thời tiết! Dữ liệu được lưu trong backend/dataset/weather_chunks/")