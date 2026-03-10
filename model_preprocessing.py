import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib
import glob

# Папки
train_dir = 'train'
test_dir = 'test'
cleared_train_dir = 'cleared_data/train'
cleared_test_dir = 'cleared_data/test'
model_dir = 'models'

# Создаём директории
os.makedirs(cleared_train_dir, exist_ok=True)
os.makedirs(cleared_test_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Признаки для масштабирования (исключаем date и target)
feature_columns = ['temperature', 'pressure', 'humidity', 'wind_speed']
target_column = 'rain'

def detect_anomalies_iqr(df, columns, multiplier=1.5):
    """
    Возвращает булев массив: True для нормальных строк, False для аномальных.
    Аномалия: значение за пределами [Q1 - m*IQR, Q3 + m*IQR] хотя бы по одному признаку.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask
    return mask

# 1. Загружаем все train файлы (кроме уже масштабированных)
train_files = glob.glob(os.path.join(train_dir, '*.csv'))
train_files = [f for f in train_files if not f.endswith('_scaled.csv')]

all_train_clean = []  # для сбора очищенных данных

for file in train_files:
    df = pd.read_csv(file)
    print(f"Обработка {file}: исходный размер {len(df)}")
    
    # Удаляем аномалии
    anomaly_mask = detect_anomalies_iqr(df, feature_columns)
    df_clean = df[anomaly_mask].copy()
    removed = len(df) - len(df_clean)
    print(f"  Удалено аномалий: {removed} ({removed/len(df)*100:.1f}%)")
    
    if len(df_clean) > 0:
        all_train_clean.append(df_clean)

# Объединяем все очищенные train данные
if all_train_clean:
    train_combined = pd.concat(all_train_clean, ignore_index=True)
    print(f"Всего тренировочных записей после очистки: {len(train_combined)}")
else:
    raise ValueError("Нет данных после очистки аномалий!")

# 2. Обучаем StandardScaler на очищенных train данных
X_train = train_combined[feature_columns]
scaler = StandardScaler()
scaler.fit(X_train)

# Сохраняем scaler
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# 3. Масштабируем очищенные train данные и сохраняем единый файл
X_train_scaled = scaler.transform(X_train)
train_scaled = train_combined.copy()
train_scaled[feature_columns] = X_train_scaled

# Сохраняем в cleared_data/train/train_scaled.csv
train_scaled.to_csv(os.path.join(cleared_train_dir, 'train_scaled.csv'), index=False)
print(f"Сохранён очищенный и масштабированный train: {cleared_train_dir}/train_scaled.csv")

# 4. Обрабатываем тестовые файлы (аномалий в них нет, но для единообразия тоже применим IQR)
test_files = glob.glob(os.path.join(test_dir, '*.csv'))
test_files = [f for f in test_files if not f.endswith('_scaled.csv')]

all_test_clean = []

for file in test_files:
    df = pd.read_csv(file)
    print(f"Обработка {file}: размер {len(df)}")
    
    # Можно также удалить возможные выбросы (хотя их быть не должно)
    anomaly_mask = detect_anomalies_iqr(df, feature_columns)
    df_clean = df[anomaly_mask].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"  В тесте удалено выбросов: {removed}")
    
    if len(df_clean) > 0:
        all_test_clean.append(df_clean)

if all_test_clean:
    test_combined = pd.concat(all_test_clean, ignore_index=True)
    print(f"Всего тестовых записей после очистки: {len(test_combined)}")
else:
    raise ValueError("Нет тестовых данных!")

# Масштабируем тестовые данные тем же scaler'ом
X_test = test_combined[feature_columns]
X_test_scaled = scaler.transform(X_test)
test_scaled = test_combined.copy()
test_scaled[feature_columns] = X_test_scaled

# Сохраняем единый файл для теста
test_scaled.to_csv(os.path.join(cleared_test_dir, 'test_scaled.csv'), index=False)
print(f"Сохранён очищенный и масштабированный test: {cleared_test_dir}/test_scaled.csv")

print("Предобработка завершена.")