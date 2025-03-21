import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

train_file = "backend/dataset/processed/train_data.csv"
test_file = "backend/dataset/processed/test_data.csv"
scaler_file = "backend/dataset/processed/scaler.pkl"
checkpoint_dir = "backend/model/checkpoints/"

latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_epoch_spatial.h5")
best_model_file = os.path.join(checkpoint_dir, "best_model_spatial.h5")
epoch_tracker_file = os.path.join(checkpoint_dir, "latest_epoch_spatial.txt")

os.makedirs(checkpoint_dir, exist_ok=True)

with open(scaler_file, 'rb') as f:
    scaler_X, scaler_y = pickle.load(f)

latitudes = np.arange(8, 25, 1)
longitudes = np.arange(102, 119, 1)
feature_cols = ["hour", "day", "month", "season", 
                "WS10M", "QV2M", "PS", "PRECTOTCORR", "T2M", "CLRSKY_SFC_SW_DWN"]
target_cols = ["CLRSKY_SFC_SW_DWN", "PS", "T2M", "QV2M", "WS10M", "PRECTOTCORR"]
timesteps = 24
batch_size = 256

def data_generator(file_path, feature_cols, target_cols, batch_size=128, timesteps=24):
    original_cols = ["Latitude", "Longitude"] + feature_cols
    for chunk in pd.read_csv(file_path, chunksize=batch_size * 5, parse_dates=["Datetime"], low_memory=True):
        chunk.sort_values(by=["Datetime"], inplace=True)
        X_scaled_full = scaler_X.transform(chunk[original_cols])
        X_scaled = X_scaled_full[:, 2:]  
        y_scaled = scaler_y.transform(chunk[target_cols])

        X_batch, lat_batch, lon_batch, y_batch = [], [], [], []
        for i in range(len(chunk) - timesteps - 24):
            lat_batch.append(chunk["Latitude"].iloc[i])
            lon_batch.append(chunk["Longitude"].iloc[i])
            X_batch.append(X_scaled[i:i+timesteps])
            y_batch.append(y_scaled[i+timesteps:i+timesteps+24])

            if len(X_batch) >= batch_size:
                yield (np.array(lat_batch), np.array(lon_batch), np.array(X_batch)), np.array(y_batch)
                X_batch, lat_batch, lon_batch, y_batch = [], [], [], []

        if X_batch:
            yield (np.array(lat_batch), np.array(lon_batch), np.array(X_batch)), np.array(y_batch)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_file, feature_cols, target_cols, batch_size=256, timesteps=24),
    output_signature=(
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, timesteps, len(feature_cols)), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 24, len(target_cols)), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_file, feature_cols, target_cols, batch_size=256, timesteps=24),
    output_signature=(
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, timesteps, len(feature_cols)), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 24, len(target_cols)), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE)

initial_epoch = 0
if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Đang tải mô hình từ checkpoint gần nhất: {latest_checkpoint_path}")
    model_spatial = load_model(latest_checkpoint_path)
    if os.path.exists(epoch_tracker_file):
        with open(epoch_tracker_file, "r") as f:
            initial_epoch = int(f.read().strip())
else:
    print("🚀 Khởi tạo mô hình mới và nạp trọng số từ mô hình cũ...")
    input_lat = Input(shape=(1,), name='latitude')
    input_lon = Input(shape=(1,), name='longitude')
    input_lstm = Input(shape=(timesteps, len(feature_cols)), name='time_series')

    embedding_lat = Embedding(input_dim=len(latitudes), output_dim=1)(input_lat)
    embedding_lon = Embedding(input_dim=len(longitudes), output_dim=1)(input_lon)

    embedding_lat_expanded = tf.repeat(embedding_lat, repeats=timesteps, axis=1)
    embedding_lon_expanded = tf.repeat(embedding_lon, repeats=timesteps, axis=1)

    merged_inputs = Concatenate(axis=-1)([embedding_lat_expanded, embedding_lon_expanded, input_lstm])
    lstm_out_1 = LSTM(128, return_sequences=True)(merged_inputs)
    lstm_out_2 = LSTM(64)(lstm_out_1)
    output = Dense(24 * len(target_cols))(lstm_out_2)
    output = Reshape((24, len(target_cols)))(output)

    model_spatial = Model(inputs=[input_lat, input_lon, input_lstm], outputs=output)

    old_model_path = os.path.join(checkpoint_dir, "best_model.h5")
    old_model = load_model(old_model_path)
    model_spatial.layers[-3].set_weights(old_model.layers[2].get_weights())

    model_spatial.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

for layer in model_spatial.layers[:6]:
    layer.trainable = False

class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        self.file_path = file_path
    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_path, "w") as f:
            f.write(str(epoch + 1))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(latest_checkpoint_path, save_best_only=False, verbose=1),
    ModelCheckpoint(best_model_file, monitor='val_loss', save_best_only=True, verbose=1),
    EpochTracker(epoch_tracker_file)
]

model_spatial.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=1
)

model_spatial.save(best_model_file)
print(f"✅ Training hoàn tất. Mô hình tốt nhất đã lưu tại {best_model_file}")
