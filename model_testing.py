import pandas as pd
import numpy as np
import tensorflow as tf
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
    df = pd.read_csv(filepath)
    
    # Проверяем наличие целевого столбца
    if 'rain' not in df.columns:
        raise ValueError("В данных отсутствует столбец 'rain'")
    
    # Отделяем признаки (X) и целевую переменную (y)
    X = df.drop('rain', axis=1)
    y = df['rain']
    
    # При необходимости преобразуем категориальные признаки в числовые
    # (если данные уже предобработаны, этот шаг можно пропустить)
    # Предполагаем, что все признаки уже числовые
    # Если есть пропуски, можно заполнить средним или удалить строки
    if X.isnull().any().any():
        print("Внимание: обнаружены пропуски в признаках. Заполняем средним.")
        X = X.fillna(X.mean())
    
    return X, y

def determine_task_type(y):
    """Определяет тип задачи: классификация или регрессия."""
    unique_values = y.nunique()
    if y.dtype == 'object' or unique_values < 20:  # вероятно, категориальный
        return 'classification'
    else:
        return 'regression'

def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """Выводит метрики для классификации."""
    # Для бинарной классификации преобразуем вероятности в метки
    if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
        # Многоклассовая
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        # Бинарная
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
    
    print("\n=== Отчёт по классификации ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
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

def evaluate_regression(y_true, y_pred):
    """Выводит метрики для регрессии."""
    y_true = y_true.values.flatten() if hasattr(y_true, 'values') else np.array(y_true).flatten()
    y_pred = y_pred.flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("\n=== Отчёт по регрессии ===")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Отклонения: разница между предсказанием и фактом
    deviations = y_pred - y_true
    print(f"\nСреднее отклонение: {deviations.mean():.4f}")
    print(f"Стандартное отклонение ошибок: {deviations.std():.4f}")
    print(f"Минимальное отклонение: {deviations.min():.4f}, максимальное: {deviations.max():.4f}")

def main():
    # 1. Проверка наличия файлов
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Ошибка: файл тестовых данных не найден по пути {TEST_DATA_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: файл модели не найден по пути {MODEL_PATH}")
        return
    
    # 2. Загрузка данных
    print("Загрузка тестовых данных...")
    X_test, y_test = load_and_prepare_data(TEST_DATA_PATH)
    print(f"Загружено {len(X_test)} образцов, {X_test.shape[1]} признаков.")
    
    # 3. Загрузка модели
    print("Загрузка модели Keras...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Модель успешно загружена.")
    
    # 4. Предсказание
    print("Выполнение предсказаний...")
    y_pred = model.predict(X_test)
    
    # 5. Определение типа задачи и оценка
    task_type = determine_task_type(y_test)
    print(f"\nОпределён тип задачи: {task_type}")
    
    if task_type == 'classification':
        evaluate_classification(y_test, y_pred)
    else:
        evaluate_regression(y_test, y_pred)

if __name__ == "__main__":
    main()