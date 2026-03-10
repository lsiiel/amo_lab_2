import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

# Фиксируем seed для воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Создаём директории
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

def generate_data(num_samples, start_date='2023-01-01', with_anomalies=False, anomaly_ratio=0.05):
    """
    Генерирует данные о погоде.
    """
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(num_samples)]
    
    # Нормальные распределения для признаков
    temperature = np.random.normal(15, 10, num_samples)
    pressure = np.random.normal(1013, 10, num_samples)
    humidity = np.random.normal(70, 20, num_samples)
    wind_speed = np.random.normal(5, 3, num_samples)
    
    # Целевая переменная (осадки) — линейная комбинация признаков + шум
    rain = (humidity - 50) * 0.1 + (pressure - 1000) * 0.05 + np.random.normal(0, 5, num_samples)
    rain = np.clip(rain, 0, None)   # осадки не могут быть отрицательными
    
    # Добавляем небольшой шум ко всем признакам
    temperature += np.random.normal(0, 0.5, num_samples)
    pressure += np.random.normal(0, 1, num_samples)
    humidity += np.random.normal(0, 2, num_samples)
    wind_speed += np.random.normal(0, 0.3, num_samples)
    
    # Внесение аномалий (если требуется)
    if with_anomalies:
        num_anomalies = int(num_samples * anomaly_ratio)
        anomaly_indices = random.sample(range(num_samples), num_anomalies)
        for idx in anomaly_indices:
            temperature[idx] = np.random.choice([-20, 40]) + np.random.normal(0, 5)
            pressure[idx] = np.random.choice([950, 1050]) + np.random.normal(0, 10)
            humidity[idx] = np.random.choice([0, 100]) + np.random.normal(0, 5)
            wind_speed[idx] = np.random.choice([0, 30]) + np.random.normal(0, 5)
            rain[idx] = np.random.choice([0, 100]) + np.random.normal(0, 10)
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'rain': rain
    })
    return df

# Параметры
TOTAL_TRAIN = 1500
TOTAL_TEST = 500

# Создаём 3 train файла по 500 записей, все с аномалиями (доля 10%)
train_sizes = [500, 500, 500]
train_files = ['train_1.csv', 'train_2.csv', 'train_3.csv']
for i, size in enumerate(train_sizes):
    # Разные начальные даты, чтобы данные не пересекались
    if i == 0:
        start_date = '2023-01-01'
    elif i == 1:
        start_date = '2023-05-01'
    else:
        start_date = '2023-09-01'
    
    df = generate_data(size, start_date=start_date, with_anomalies=True, anomaly_ratio=0.1)
    df.to_csv(os.path.join('train', train_files[i]), index=False)

# Создаём 3 test файла (суммарно 500 записей), без аномалий
test_sizes = [200, 150, 150]
test_files = ['test_1.csv', 'test_2.csv', 'test_3.csv']
for i, size in enumerate(test_sizes):
    if i == 0:
        start_date = '2024-01-01'
    elif i == 1:
        start_date = '2024-03-01'
    else:
        start_date = '2024-06-01'
    
    df = generate_data(size, start_date=start_date, with_anomalies=False)
    df.to_csv(os.path.join('test', test_files[i]), index=False)

print("Data generation complete. Files saved:")
print("Train files (with anomalies):")
for f in train_files:
    print(f"  train/{f}")
print("Test files (without anomalies):")
for f in test_files:
    print(f"  test/{f}")