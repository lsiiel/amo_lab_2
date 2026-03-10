#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Запуск пайплайна ===${NC}"

# Проверка наличия Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo -e "${RED}Ошибка: Python не найден. Установите Python и добавьте в PATH.${NC}"
    exit 1
fi

echo "Используется Python: $($PYTHON --version)"

# Создание виртуального окружения, если его нет
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    $PYTHON -m venv venv
fi

# Активация окружения (для Windows)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OS" == "Windows_NT" ]]; then
    echo "Активация виртуального окружения для Windows..."
    source venv/Scripts/activate
else
    echo "Активация виртуального окружения для Unix..."
    source venv/bin/activate
fi

echo "Виртуальное окружение активировано."

# Установка зависимостей
if [ -f "requirements.txt" ]; then
    echo "Установка зависимостей..."
    #Загрузка с китайского сервера для скорости (по умолчанию гораздо медленнее)
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
else
    echo -e "${RED}Предупреждение: файл requirements.txt не найден.${NC}"
fi

# Запуск скриптов
echo -e "${GREEN}--- Шаг 1: data_creation.py ---${NC}"
python data_creation.py

echo -e "${GREEN}--- Шаг 2: model_preprocessing.py ---${NC}"
python model_preprocessing.py

echo -e "${GREEN}--- Шаг 3: model_preparation.py ---${NC}"
python model_preparation.py

echo -e "${GREEN}--- Шаг 4: model_testing.py ---${NC}"
python model_testing.py

echo -e "${GREEN}=== Пайплайн успешно завершён! ===${NC}"

# Деактивация окружения
deactivate