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
    
    print(f"Размер данных: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print(f"Первые 5 строк:")
    print(df.head())
    
    # Проверяем наличие целевого столбца
    if 'rain' not in df.columns:
        raise ValueError("В данных отсутствует столбец 'rain'")
    
    # Исключаем нечисловые колонки (date) и целевую переменную
    # Берем только числовые колонки для признаков
    X = df.select_dtypes(include=[np.number]).drop('rain', axis=1, errors='ignore')
    y = df['rain']
    
    print(f"Используемые признаки: {list(X.columns)}")
    print(f"Форма X: {X.shape}")
    print(f"Форма y: {y.shape}")
    
    # Преобразуем в numpy массивы
    X = X.values.astype('float32')
    y = y.values.astype('float32')
    
    return X, y

def determine_task_type(y):
    """Определяет тип задачи: классификация или регрессия."""
    unique_values = len(np.unique(y))
    print(f"Уникальных значений в целевой переменной: {unique_values}")
    
    # Если все значения целые и их мало - классификация
    if np.all(y == y.astype(int)) and unique_values < 20:
        return 'classification'
    else:
        return 'regression'

def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """Выводит метрики для классификации."""
    if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
        y_pred = np.argmax(y_pred_prob, axis=1)
        print("Многоклассовая классификация")
    else:
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        print("Бинарная классификация")
    
    print("\n" + "="*50)
    print("=== ОТЧЁТ ПО КЛАССИФИКАЦИИ ===")
    print("="*50)
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_true, y_pred))
    
    errors = (y_true != y_pred).sum()
    print(f"\nВсего ошибок: {errors} из {len(y_true)} ({errors/len(y_true)*100:.2f}%)")
    print("="*50)

def evaluate_regression(y_true, y_pred):
    """Выводит метрики для регрессии."""
    y_true = y_true.flatten()
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
    
    deviations = y_pred - y_true
    print(f"\nСреднее отклонение: {deviations.mean():.4f}")
    print(f"Стандартное отклонение ошибок: {deviations.std():.4f}")
    
    abs_deviations = np.abs(deviations)
    print(f"\nСреднее абсолютное отклонение: {abs_deviations.mean():.4f}")
    print(f"Медианное абсолютное отклонение: {np.median(abs_deviations):.4f}")
    print(f"95-й перцентиль ошибок: {np.percentile(abs_deviations, 95):.4f}")
    print("="*50)

def main():
    global TEST_DATA_PATH, MODEL_PATH
    
    print("\n" + "="*60)
    print("ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*60)
    
    # 1. Проверка наличия файлов
    print(f"\nПроверка файлов:")
    print(f"Тестовые данные: {TEST_DATA_PATH} - {os.path.exists(TEST_DATA_PATH)}")
    print(f"Модель: {MODEL_PATH} - {os.path.exists(MODEL_PATH)}")
    
    if not os.path.exists(TEST_DATA_PATH):
        alt_path = "cleared_data/test/test.csv"
        if os.path.exists(alt_path):
            print(f"Найден альтернативный файл: {alt_path}")
            TEST_DATA_PATH = alt_path
        else:
            print(f"Ошибка: файл тестовых данных не найден")
            return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: файл модели не найден по пути {MODEL_PATH}")
        return
    
    # 2. Загрузка данных
    print("\n" + "-"*40)
    print("ШАГ 1: Загрузка тестовых данных")
    print("-"*40)
    try:
        X_test, y_test = load_and_prepare_data(TEST_DATA_PATH)
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
        print(f"Ожидаемая форма входа: {model.input_shape}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return
    
    # 4. Проверка соответствия признаков
    print("\n" + "-"*40)
    print("ШАГ 3: Проверка признаков")
    print("-"*40)
    expected_features = model.input_shape[1]
    print(f"Модель ожидает: {expected_features} признаков")
    print(f"Тестовые данные: {X_test.shape[1]} признаков")
    
    if X_test.shape[1] != expected_features:
        print(f"⚠ ВНИМАНИЕ: Количество признаков не совпадает!")
        print(f"Используем только первые {expected_features} признаков")
        X_test = X_test[:, :expected_features]
        print(f"Новая форма X_test: {X_test.shape}")
    
    # 5. Предсказание
    print("\n" + "-"*40)
    print("ШАГ 4: Выполнение предсказаний")
    print("-"*40)
    try:
        y_pred = model.predict(X_test)
        print(f"Форма предсказаний: {y_pred.shape}")
        print(f"Пример предсказаний (первые 5):\n{y_pred[:5].flatten()}")
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return
    
    # 6. Оценка модели
    print("\n" + "-"*40)
    print("ШАГ 5: Оценка модели")
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