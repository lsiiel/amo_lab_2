import pandas as pd
import numpy as np
import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import os

# Параметры
TEST_DATA_PATH = "cleared_data/test/test_scaled.csv"
MODEL_PATH = "models/model.keras"

def load_and_prepare_data(filepath):
    """Загружает тестовые данные и разделяет на признаки и целевую переменную."""
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Первые 5 строк данных:")
    print(df.head())
    print(f"\nИнформация о данных:")
    print(df.info())
    
    # Проверяем наличие целевого столбца
    if 'rain' not in df.columns:
        raise ValueError("В данных отсутствует столбец 'rain'")
    
    # Отделяем признаки (X) и целевую переменную (y)
    X = df.drop('rain', axis=1)
    y = df['rain']
    
    print(f"\nТипы данных ДО преобразования:")
    print(X.dtypes.value_counts())
    print(f"Object колонки: {list(X.select_dtypes(include=['object']).columns)}")
    
    # ПРЕОБРАЗОВАНИЕ ТИПОВ
    # 1. Сначала преобразуем object columns в числовые
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"\nНайдены object-колонки: {list(object_cols)}")
        for col in object_cols:
            print(f"Преобразование колонки {col}...")
            # Пробуем преобразовать в числа, нечисловые станут NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 2. Преобразуем все колонки в числовые типы
    for col in X.columns:
        if X[col].dtype == 'object':
            # Если после to_numeric остались object, используем one-hot encoding
            print(f"Колонка {col} требует one-hot encoding")
            X = pd.get_dummies(X, columns=[col], prefix=[col])
        else:
            # Конвертируем в float32
            X[col] = X[col].astype('float32')
    
    print(f"\nТипы данных ПОСЛЕ преобразования:")
    print(X.dtypes.value_counts())
    
    # Обработка пропусков
    if X.isnull().any().any():
        print("\nВнимание: обнаружены пропуски в признаках.")
        null_counts = X.isnull().sum()
        print(f"Пропуски по колонкам:\n{null_counts[null_counts > 0]}")
        
        # Заполняем пропуски средним по колонке
        for col in X.columns:
            if X[col].isnull().any():
                mean_val = X[col].mean()
                if pd.isna(mean_val):  # если среднее тоже NaN
                    mean_val = 0
                X[col] = X[col].fillna(mean_val)
                print(f"Колонка {col}: заполнено {X[col].isnull().sum()} пропусков средним {mean_val:.4f}")
    
    # Финальная проверка
    print(f"\nФинальная форма X: {X.shape}")
    print(f"Финальные типы: {X.dtypes.unique()}")
    print(f"Есть ли NaN: {X.isnull().any().any()}")
    
    return X, y

def determine_task_type(y):
    """Определяет тип задачи: классификация или регрессия."""
    unique_values = y.nunique()
    print(f"Уникальных значений в целевой переменной: {unique_values}")
    print(f"Тип целевой переменной: {y.dtype}")
    
    if y.dtype == 'object' or unique_values < 20:
        return 'classification'
    else:
        return 'regression'

def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """Выводит метрики для классификации."""
    # Для бинарной классификации преобразуем вероятности в метки
    if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
        # Многоклассовая
        y_pred = np.argmax(y_pred_prob, axis=1)
        print("Многоклассовая классификация")
    else:
        # Бинарная
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        print("Бинарная классификация")
    
    print("\n" + "="*50)
    print("=== ОТЧЁТ ПО КЛАССИФИКАЦИИ ===")
    print("="*50)
    
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nДетальный отчёт:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Отклонения: количество неправильных предсказаний
    errors = (y_true != y_pred).sum()
    print(f"\nВсего ошибок: {errors} из {len(y_true)} ({errors/len(y_true)*100:.2f}%)")
    print("="*50)

def evaluate_regression(y_true, y_pred):
    """Выводит метрики для регрессии."""
    y_true = y_true.values.flatten() if hasattr(y_true, 'values') else np.array(y_true).flatten()
    y_pred = y_pred.flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("=== ОТЧЁТ ПО РЕГРЕССИИ ===")
    print("="*50)
    
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Отклонения: разница между предсказанием и фактом
    deviations = y_pred - y_true
    print(f"\nСреднее отклонение: {deviations.mean():.4f}")
    print(f"Стандартное отклонение ошибок: {deviations.std():.4f}")
    print(f"Минимальное отклонение: {deviations.min():.4f}")
    print(f"Максимальное отклонение: {deviations.max():.4f}")
    
    # Дополнительная статистика
    abs_deviations = np.abs(deviations)
    print(f"\nСреднее абсолютное отклонение: {abs_deviations.mean():.4f}")
    print(f"Медианное абсолютное отклонение: {np.median(abs_deviations):.4f}")
    print(f"95-й перцентиль ошибок: {np.percentile(abs_deviations, 95):.4f}")
    print("="*50)

def main():
    # Используем глобальные переменные
    global TEST_DATA_PATH, MODEL_PATH
    
    print("\n" + "="*60)
    print("ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*60)
    
    # 1. Проверка наличия файлов
    print(f"\nПроверка файлов:")
    print(f"Тестовые данные: {TEST_DATA_PATH} - {os.path.exists(TEST_DATA_PATH)}")
    print(f"Модель: {MODEL_PATH} - {os.path.exists(MODEL_PATH)}")
    
    # Создаем локальную копию для возможных изменений
    test_data_path = TEST_DATA_PATH
    
    if not os.path.exists(test_data_path):
        print(f"Ошибка: файл тестовых данных не найден по пути {test_data_path}")
        # Попробуем найти альтернативный путь
        alt_path = "cleared_data/test/test.csv"
        if os.path.exists(alt_path):
            print(f"Найден альтернативный файл: {alt_path}")
            test_data_path = alt_path
        else:
            print("Альтернативный файл тоже не найден")
            return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: файл модели не найден по пути {MODEL_PATH}")
        return
    
    # 2. Загрузка данных
    print("\n" + "-"*40)
    print("ШАГ 1: Загрузка тестовых данных")
    print("-"*40)
    try:
        X_test, y_test = load_and_prepare_data(test_data_path)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # 3. Загрузка модели
    print("\n" + "-"*40)
    print("ШАГ 2: Загрузка модели Keras")
    print("-"*40)
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Модель успешно загружена.")
        print(f"Архитектура модели:")
        model.summary()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return
    
    # 4. Предсказание
    print("\n" + "-"*40)
    print("ШАГ 3: Выполнение предсказаний")
    print("-"*40)
    try:
        y_pred = model.predict(X_test)
        print(f"Форма предсказаний: {y_pred.shape}")
        print(f"Тип предсказаний: {y_pred.dtype}")
        print(f"Пример предсказаний (первые 5):\n{y_pred[:5]}")
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return
    
    # 5. Определение типа задачи и оценка
    print("\n" + "-"*40)
    print("ШАГ 4: Оценка модели")
    print("-"*40)
    task_type = determine_task_type(y_test)
    print(f"Определён тип задачи: {task_type}")
    
    if task_type == 'classification':
        evaluate_classification(y_test, y_pred)
    else:
        evaluate_regression(y_test, y_pred)
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)

if __name__ == "__main__":
    main()