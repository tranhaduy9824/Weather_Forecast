import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import gc

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

print("🖥️ Các thiết bị khả dụng:", tf.config.list_physical_devices('GPU'))

train_file = "backend/dataset/processed/train_data.csv"
test_file = "backend/dataset/processed/test_data.csv"
scaler_file = "backend/dataset/processed/scaler.pkl"
checkpoint_dir = "backend/model/checkpoints/"
fine_tuned_model_file = "backend/model/cnn_lstm_weather.h5"
best_model_file = os.path.join(checkpoint_dir, "best_model_spatial.h5")
epoch_file = os.path.join(checkpoint_dir, "last_epoch.txt")

os.makedirs(checkpoint_dir, exist_ok=True)

feature_cols = ["Latitude", "Longitude", "hour", "day", "month", "season", 
                "WS10M", "QV2M", "PS", "PRECTOTCORR", "T2M", "CLRSKY_SFC_SW_DWN"]
target_cols = ["CLRSKY_SFC_SW_DWN", "PS", "T2M", "QV2M", "WS10M", "PRECTOTCORR"] 

dtype_dict = {col: np.float32 for col in feature_cols + target_cols}
dtype_dict.update({"hour": np.int8, "day": np.int8, "month": np.int8, "season": np.int8})

with open(scaler_file, 'rb') as f:
    scaler_X, scaler_y = pickle.load(f)

best_model_path = os.path.join(checkpoint_dir, "best_model.h5")
print(f"🔄 Đang tải model từ: {best_model_path}")
old_model = load_model(best_model_path)

latitudes = np.arange(8, 25, 1) 
longitudes = np.arange(102, 119, 1) 
locations = [(lat, lon) for lat in latitudes for lon in longitudes]
print(f"✅ Tổng số tọa độ sẽ được huấn luyện: {len(locations)}")

def data_generator_multi(file_path, feature_cols, target_cols, batch_size=256, timesteps=24, num_locations=5):
    for chunk in pd.read_csv(file_path, chunksize=batch_size * num_locations, dtype=dtype_dict, parse_dates=["Datetime"], low_memory=True):
        chunk = chunk.sort_values(by=["Datetime", "Latitude", "Longitude"])

        X_scaled = scaler_X.transform(chunk[feature_cols])
        y_scaled = scaler_y.transform(chunk[target_cols])

        X_batch, y_batch = [], []
        if len(chunk) < timesteps + 24:
            continue

        for i in range(0, len(chunk) - timesteps - 24 + 1, num_locations):
            batch_data = X_scaled[i:i + timesteps].reshape(timesteps, len(feature_cols))
            target_data = y_scaled[i + timesteps:i + timesteps + 24]
            if batch_data.shape[0] == timesteps and target_data.shape[0] == 24:
                X_batch.append(batch_data)
                y_batch.append(target_data)

            if len(X_batch) >= batch_size:
                yield np.array(X_batch), np.array(y_batch)
                X_batch, y_batch = [], []

        if X_batch: 
            yield np.array(X_batch), np.array(y_batch)

batch_size = 256  
timesteps = 24  

train_dataset_multi = tf.data.Dataset.from_generator(
    lambda: data_generator_multi(train_file, feature_cols, target_cols, batch_size, timesteps),
    output_signature=(
        tf.TensorSpec(shape=(None, timesteps, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 24, len(target_cols)), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE)

start_epoch = 0
if os.path.exists(epoch_file):
    with open(epoch_file, "r") as f:
        start_epoch = int(f.read())
    print(f"🔄 Tiếp tục huấn luyện từ epoch: {start_epoch}")

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', secondary_monitor='val_mae', save_best_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.secondary_monitor = secondary_monitor
        self.save_best_only = save_best_only
        self.best_loss = np.inf
        self.best_mae = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        current_mae = logs.get(self.secondary_monitor)

        if current_loss is None or current_mae is None:
            return

        if self.save_best_only:
            if current_loss < self.best_loss or (current_loss == self.best_loss and current_mae < self.best_mae):
                self.best_loss = current_loss
                self.best_mae = current_mae
                self.model.save(self.filepath, overwrite=True)
                print(f"\n💾 Lưu mô hình tốt nhất: {self.monitor} = {current_loss:.4f}, {self.secondary_monitor} = {current_mae:.4f}")
        else:
            self.model.save(self.filepath, overwrite=True)

print("🚀 Fine-tune mô hình cũ để học quan hệ giữa các tọa độ...")
try:
    history = old_model.fit(
        train_dataset_multi,
        epochs=10,
        initial_epoch=start_epoch,
        validation_data=train_dataset_multi,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            CustomModelCheckpoint(best_model_file, monitor='val_loss', secondary_monitor='val_mae', save_best_only=True)
        ],
        verbose=1
    )
    with open(epoch_file, "w") as f:
        f.write(str(history.epoch[-1] + 1))
except KeyboardInterrupt:
    print("\n⛔ Đã dừng huấn luyện bởi Ctrl+C")
    current_epoch = start_epoch + len(history.epoch)
    with open(epoch_file, "w") as f:
        f.write(str(current_epoch))
    old_model.save(best_model_file)
    print(f"✅ Đã lưu epoch gần nhất ({current_epoch}) vào {epoch_file} và mô hình vào {best_model_file}")
    exit(0)

old_model.save(fine_tuned_model_file)
print("✅ Mô hình đã được fine-tune và lưu lại.")

print("🚀 Thêm CNN để mô hình học mạnh hơn về quan hệ không gian...")
cnn_lstm_model = Sequential([
    Input(shape=(timesteps, len(feature_cols) * 5)),

    Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
    Flatten(),

    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),

    Dense(24 * len(target_cols)),
    tf.keras.layers.Reshape((24, len(target_cols))) 
])

cnn_lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

for layer in old_model.layers:
    layer.trainable = False

try:
    history = cnn_lstm_model.fit(
        train_dataset_multi,
        epochs=10,
        initial_epoch=start_epoch,
        validation_data=train_dataset_multi,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            CustomModelCheckpoint(best_model_file, monitor='val_loss', secondary_monitor='val_mae', save_best_only=True)
        ],
        verbose=1
    )
    with open(epoch_file, "w") as f:
        f.write(str(history.epoch[-1] + 1))
except KeyboardInterrupt:
    print("\n⛔ Đã dừng huấn luyện bởi Ctrl+C")
    current_epoch = start_epoch + len(history.epoch)
    with open(epoch_file, "w") as f:
        f.write(str(current_epoch))
    cnn_lstm_model.save(best_model_file)
    print(f"✅ Đã lưu epoch gần nhất ({current_epoch}) vào {epoch_file} và mô hình vào {best_model_file}")
    exit(0)

cnn_lstm_model.save("backend/model/checkpoints/best_model_spatial.h5")
print("✅ Mô hình CNN-LSTM đã được fine-tune và lưu lại.")
