import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dog_breed_detector.bot.bot import main

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Запускаем бота
    main()