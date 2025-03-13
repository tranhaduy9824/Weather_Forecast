import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import pickle
import os
import gc
import glob

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

print("🖥️ Các thiết bị khả dụng:", tf.config.list_physical_devices('GPU'))

train_file = "backend/dataset/processed/train_data.csv"
test_file = "backend/dataset/processed/test_data.csv"
scaler_file = "backend/dataset/processed/scaler.pkl"
model_checkpoint_dir = "backend/model/checkpoints/"
model_file = "backend/model/lstm_weather.h5"
log_dir = "backend/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
csv_log_file = "backend/logs/training_log.csv"

os.makedirs(model_checkpoint_dir, exist_ok=True)

feature_cols = ["Latitude", "Longitude", "hour", "day", "month", "season", "WS10M", "QV2M", "PS", "PRECTOTCORR", "T2M"]
target_cols = ["CLRSKY_SFC_SW_DWN", "PS", "T2M", "QV2M", "WS10M", "PRECTOTCORR"]

df_sample = pd.read_csv(train_file, nrows=10000, usecols=feature_cols + target_cols, on_bad_lines='skip')

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(df_sample[feature_cols])
scaler_y.fit(df_sample[target_cols])

with open(scaler_file, 'wb') as f:
    pickle.dump((scaler_X, scaler_y), f)

del df_sample 
gc.collect()

def data_generator(file_path, feature_cols, target_cols, batch_size=8192):
    for chunk in pd.read_csv(
        file_path, chunksize=batch_size, usecols=feature_cols + target_cols, on_bad_lines='skip', low_memory=True
    ):
        try:
            X_chunk = scaler_X.transform(chunk[feature_cols])
            y_chunk = scaler_y.transform(chunk[target_cols])

            X_chunk = X_chunk.reshape((X_chunk.shape[0], 1, X_chunk.shape[1]))

            yield X_chunk, y_chunk
        except Exception as e:
            print(f"⚠️ Bỏ qua lỗi khi đọc dữ liệu: {e}")
            continue

batch_size = 8192
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_file, feature_cols, target_cols, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(target_cols)), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_file, feature_cols, target_cols, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(target_cols)), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE)

latest_checkpoint = None
checkpoints = glob.glob(os.path.join(model_checkpoint_dir, "ckpt_*"))
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

with tf.device('/GPU:0'):
    if latest_checkpoint:
        print(f"🔄 Phát hiện checkpoint: {latest_checkpoint}, tiếp tục huấn luyện...")
        model = load_model(latest_checkpoint) 
        initial_epoch = int(os.path.basename(latest_checkpoint).split("_")[-1]) 
        print(f"🔄 Tiếp tục từ epoch {initial_epoch + 1}")
    else:
        print("🚀 Không tìm thấy checkpoint, bắt đầu huấn luyện từ đầu...")
        initial_epoch = 0

        model = Sequential([
            LSTM(96, return_sequences=True, input_shape=(1, len(feature_cols))),
            Dropout(0.3),
            LSTM(96, kernel_regularizer=l2(0.001)), 
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)), 
            Dense(len(target_cols))
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("🚀 Bắt đầu huấn luyện trên GPU...")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_checkpoint_dir, "ckpt_{epoch}"),
        save_best_only=False,
        save_freq="epoch",
        monitor='val_loss',
        mode='min'
    )
    csv_logger = CSVLogger(csv_log_file, append=True) 

    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            print(f"🧹 Dọn dẹp bộ nhớ sau epoch {epoch+1}")

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )

    model.fit(
        train_dataset,
        epochs=50,
        initial_epoch=initial_epoch,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, checkpoint_callback, MemoryCleanupCallback(), csv_logger, early_stopping, reduce_lr]
    )

    model.save(model_file)

print("✅ Mô hình LSTM đã được huấn luyện và lưu lại trên GPU.")
