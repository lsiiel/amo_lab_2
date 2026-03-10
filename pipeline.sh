#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Запуск пайплайна ===${NC}"

# Установка зависимостей (если не установлены в образе)
if [ -f "requirements.txt" ]; then
    echo "Установка зависимостей..."
    pip install --upgrade pip
    pip install -r requirements.txt
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