import os
import keras
import numpy as np
import pandas as pd
from utils import get_model_path
from keras import layers
from sklearn.preprocessing import StandardScaler
from constants import MODEL_FOLDER, MODEL_NAME, TARGET, TEST_DATAFRAME, TRAIN_DATAFRAME

train = pd.read_csv(TRAIN_DATAFRAME)
test = pd.read_csv(TEST_DATAFRAME)

X_train = train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).values
y_train = train[TARGET].values

X_test = test.drop(columns=[TARGET]).select_dtypes(include=[np.number]).values
y_test = test[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[cb])

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nMAE: {mae:.2f} мм")

y_pred = model.predict(X_test).flatten()
df = pd.DataFrame({
    'Факт, мм': y_test[:5],
    'Прогноз, мм': y_pred[:5]
})
print(df.to_string(float_format='%.1f', index=False))

model_path = get_model_path(MODEL_FOLDER, MODEL_NAME)

os.makedirs(MODEL_FOLDER, exist_ok=True)
model.save(model_path)
