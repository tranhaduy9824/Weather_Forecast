import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import gc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

print("🖥️ Các thiết bị khả dụng:", tf.config.list_physical_devices('GPU'))

# Cấu hình đường dẫn
train_file = "backend/dataset/processed/train_data2.csv"
test_file = "backend/dataset/processed/test_data2.csv"
scaler_file = "backend/dataset/processed/scaler_new.pkl"
checkpoint_dir = "backend/model/checkpoints_new/"
model_file = "backend/model/lstm_weather.h5"

os.makedirs(checkpoint_dir, exist_ok=True)

# Cột đặc trưng và mục tiêu
feature_cols = ["Latitude", "Longitude", "hour", "day", "month", "season", 
                "WS10M", "QV2M", "PS", "PRECTOTCORR", "T2M", "CLRSKY_SFC_SW_DWN"]
target_cols = ["CLRSKY_SFC_SW_DWN", "PS", "T2M", "QV2M", "WS10M", "PRECTOTCORR"]

dtype_dict = {col: np.float32 for col in feature_cols + target_cols}
dtype_dict.update({"hour": np.int8, "day": np.int8, "month": np.int8, "season": np.int8})

# Khởi tạo scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
chunk_size = 5000

print("⚡ Bắt đầu chuẩn hóa dữ liệu toàn bộ...")
for chunk in pd.read_csv(train_file, chunksize=chunk_size, usecols=feature_cols + target_cols, dtype=dtype_dict):
    scaler_X.partial_fit(chunk[feature_cols])
    scaler_y.partial_fit(chunk[target_cols])

with open(scaler_file, 'wb') as f:
    pickle.dump((scaler_X, scaler_y), f)

del chunk
gc.collect()
print("✅ Hoàn tất chuẩn hóa!")

# Data generator cho LSTM
def data_generator(file_path, feature_cols, target_cols, batch_size=256, timesteps=24):
    for chunk in pd.read_csv(file_path, chunksize=batch_size * 10, dtype=dtype_dict, parse_dates=["Datetime"]):
        chunk = chunk.sort_values(by=["Datetime"])
        
        X_scaled = scaler_X.transform(chunk[feature_cols])
        y_scaled = scaler_y.transform(chunk[target_cols])

        num_samples = len(chunk) - timesteps - 24 + 1
        if num_samples <= 0:
            continue

        X_batch = np.zeros((num_samples, timesteps, len(feature_cols)), dtype=np.float32)
        y_batch = np.zeros((num_samples, 24, len(target_cols)), dtype=np.float32)

        for i in range(num_samples):
            X_batch[i] = X_scaled[i:i+timesteps]
            y_batch[i] = y_scaled[i+timesteps:i+timesteps+24]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield X_batch[start:end], y_batch[start:end]

# Cấu hình
batch_size = 128
timesteps = 24

# Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_file, feature_cols, target_cols, batch_size, timesteps),
    output_signature=(
        tf.TensorSpec(shape=(None, timesteps, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 24, len(target_cols)), dtype=tf.float32),
    )
).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_file, feature_cols, target_cols, batch_size, timesteps),
    output_signature=(
        tf.TensorSpec(shape=(None, timesteps, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 24, len(target_cols)), dtype=tf.float32),
    )
).cache().prefetch(tf.data.AUTOTUNE)

# Kiểm tra checkpoint
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_epoch.h5")
epoch_tracker_file = os.path.join(checkpoint_dir, "latest_epoch.txt")

if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Đang tải model từ checkpoint gần nhất: {latest_checkpoint_path}")
    model = load_model(latest_checkpoint_path)
    initial_epoch = int(open(epoch_tracker_file, "r").read().strip()) if os.path.exists(epoch_tracker_file) else 0
else:
    print("🚀 Khởi tạo model mới...")
    initial_epoch = 0
    model = Sequential([
        Input(shape=(timesteps, len(feature_cols))),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(24 * len(target_cols)),
        tf.keras.layers.Reshape((24, len(target_cols)))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='mse', 
                metrics=['mae'])

# Callbacks
class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, best_model_path):
        super().__init__()
        self.best_model_path = best_model_path
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save(self.best_model_path)
            print(f"✅ Lưu mô hình tốt nhất (val_loss={val_loss:.6f})")

best_checkpoint = CustomCheckpoint(os.path.join(checkpoint_dir, "best_model.h5"))
latest_checkpoint = ModelCheckpoint(filepath=latest_checkpoint_path, save_best_only=False, verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, epoch_file):
        super().__init__()
        self.epoch_file = epoch_file

    def on_epoch_end(self, epoch, logs=None):
        with open(self.epoch_file, "w") as f:
            f.write(str(epoch + 1))

epoch_tracker = EpochTracker(epoch_tracker_file)

# Huấn luyện
history = model.fit(
    train_dataset,
    epochs=50,
    initial_epoch=initial_epoch,
    validation_data=test_dataset,
    callbacks=[early_stopping, reduce_lr, best_checkpoint, latest_checkpoint, epoch_tracker],
    verbose=1
)

# Lưu mô hình
model.save(model_file, save_format="h5")
print("✅ Mô hình LSTM đã được huấn luyện và lưu lại.")
