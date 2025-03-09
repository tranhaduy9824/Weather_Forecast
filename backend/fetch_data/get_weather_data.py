import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from itertools import product

latitudes = np.arange(8, 25, 1)
longitudes = np.arange(102, 119, 1)

start_year = 2022
end_year = 2025
end_month = 3

chunk_folder = "backend/dataset/weather_chunks"
os.makedirs(chunk_folder, exist_ok=True)

def fetch_weather_data(lat, lon):
    output_file = f"{chunk_folder}/weather_{lat}_{lon}.csv"

    if os.path.exists(output_file):
        print(f"✅ Bỏ qua {lat}, {lon}, đã có dữ liệu.")
        return

    weather_data = []
    max_retries = 5  

    for year in range(start_year, end_year + 1):
        start_date = f"{year}0101"
        end_date = f"{year}{end_month:02d}31" if year == end_year else f"{year}1231"

        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            "parameters": "T2M,QV2M,PS,WS10M,PRECTOT,CLRSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                if "properties" in data and "parameter" in data["properties"]:
                    parameters_data = data["properties"]["parameter"]
                    print(f"📊 API trả về dữ liệu cho {lat}, {lon}, năm {year}: {parameters_data.keys()}")

                    if not parameters_data:
                        print(f"⚠️ API không trả dữ liệu hợp lệ cho {lat}, {lon}, năm {year}. Bỏ qua.")
                        return
                    
                    dates = list(parameters_data.get("T2M", {}).keys())

                    for date in dates:
                        for hour in range(24):
                            corrected_date = date[:8] 
                            row = [f"{corrected_date} {hour:02d}:00", lat, lon]

                            for param in parameters_data.keys():
                                values = parameters_data.get(param, {}).get(date, [])
                                value = values[hour] if isinstance(values, list) and hour < len(values) else values
                                row.append(value)

                            weather_data.append(row)

                    print(f"✅ Lấy dữ liệu thành công: {lat}, {lon}, năm {year}")
                    break  

            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                retry_count += 1
                print(f"⏳ Lỗi API tại {lat}, {lon}, năm {year}: {e}. Thử lại {retry_count}/{max_retries}...")
                time.sleep(3)

        if retry_count == max_retries:
            print(f"❌ Bỏ qua {lat}, {lon}, năm {year} sau {max_retries} lần thử.")

        time.sleep(0.5)

    if weather_data:
        print(f"📂 Đang lưu file: {output_file} với {len(weather_data)} dòng dữ liệu.")
        try:
            df_weather = pd.DataFrame(weather_data, columns=["Datetime", "Latitude", "Longitude"] + list(parameters_data.keys()))
            df_weather["Datetime"] = pd.to_datetime(df_weather["Datetime"], format='%Y%m%d %H:%M') 
            df_weather.sort_values(by=["Datetime", "Latitude", "Longitude"], ascending=[True, True, True], inplace=True)
            df_weather.to_csv(output_file, index=False, sep=",", float_format="%.2f")
            print(f"✅ Dữ liệu đã lưu thành công tại {output_file}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file {output_file}: {e}")
    else:
        print(f"⚠️ Không có dữ liệu hợp lệ cho {lat}, {lon}. File sẽ không được tạo.")

lat_lon_pairs = list(product(latitudes, longitudes))

with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(lambda pair: fetch_weather_data(pair[0], pair[1]), lat_lon_pairs)

print("✅ Hoàn thành lấy dữ liệu thời tiết! Dữ liệu được lưu trong backend/dataset/weather_chunks/")
