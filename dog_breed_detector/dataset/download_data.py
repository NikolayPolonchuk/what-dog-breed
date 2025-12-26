import hydra
from omegaconf import DictConfig
import zipfile
from pathlib import Path
import kaggle

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    download_data(cfg)

def download_data(cfg: DictConfig):
    """
    Скачивает датасет с Kaggle
    """
    
    # Название соревнования
    competition_name = cfg.dataset.paths.name
    
    # Путь к текущему файлу
    current_file = Path(__file__).resolve()
    
    # Поднимаемся на 2 уровня выше
    base_dir = current_file.parent.parent.parent
    
    data_dir = base_dir / cfg.dataset.paths.data_dir
    
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Папка {data_dir} уже существует и не пуста.")
        print("Скачивание не требуется.")
        return data_dir

    # Создаем папку data
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Скачивание датасета: {competition_name}")
    
    # Скачиваем архив
    kaggle.api.competition_download_files(
        competition=competition_name,
        path=str(data_dir),
        quiet=False
    )
    
    # Путь к архиву
    zip_path = data_dir / f"{competition_name}.zip"
    
    # Распаковываем
    print("Распаковка архива...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Удаляем архив если нужно
    if cfg.dataset.paths.remove_zip:
        zip_path.unlink()
        print("Архив удален")
    
    print(f"Данные сохранены в: {data_dir}")
    
    return data_dir


if __name__ == "__main__":
    main()