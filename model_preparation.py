import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("cleared_data/train/train_scaled.csv")
test = pd.read_csv("cleared_data/test/test_scaled.csv")

TARGET = "rain"

X_train = train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).values
y_train = train[TARGET].values

X_test = test.drop(columns=[TARGET]).select_dtypes(include=[np.number]).values
y_test = test[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(1, input_shape=(X_train.shape[1],))
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nMAE: {mae:.2f} мм")

y_pred = model.predict(X_test).flatten()
df = pd.DataFrame({
    'Факт, мм': y_test[:5],
    'Прогноз, мм': y_pred[:5]
})
print(df.to_string(float_format='%.1f', index=False))
