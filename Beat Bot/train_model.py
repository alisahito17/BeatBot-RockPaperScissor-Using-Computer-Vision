import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import joblib

DATASET_PATH = "hand_landmarks_dataset.csv"
data = pd.read_csv(DATASET_PATH)
X = data.iloc[:, 1:].values
y = data['label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

landmark_model = models.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

landmark_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = landmark_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

landmark_model.save("hand_landmark_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")
print("Model saved as 'hand_landmark_model.h5'")
