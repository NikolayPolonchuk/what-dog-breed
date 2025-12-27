import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset


class DogDataset(Dataset):
    def __init__(self, img_path, csv_path=None, transform=None):
        self.img_path = Path(img_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.transform = transform
        
        # Используем glob с Path
        self.img_names = list(self.img_path.glob("*.jpg"))
        
        if csv_path:
            self._load_labels()
    
    def _load_labels(self):
        label_df = pd.read_csv(self.csv_path)
        self.label_idx2name = label_df['breed'].unique()
        self.label_name2idx = {name: idx for idx, name in enumerate(self.label_idx2name)}
        
        self.img2label = {}
        for _, row in label_df.iterrows():
            # Создаем Path объект для пути к изображению
            img_path = self.img_path / f"{row['id']}.jpg"
            # Сохраняем как строку для совместимости
            self.img2label[str(img_path)] = self.label_name2idx[row['breed']]
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        # Получаем Path объект
        img_path_obj = self.img_names[index]
        # Конвертируем в строку для поиска в словаре
        img_path_str = str(img_path_obj)
        
        if self.csv_path:
            label = self.img2label[img_path_str]
            label = torch.tensor(label)
        else:
            label = -1
        
        # Используем Path объект для открытия файла
        img = Image.open(img_path_obj).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DogDataModule(pl.LightningDataModule):
    """DataModule для датасета собак"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_transform = None
        self.val_transform = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.channel_mean = None
        self.channel_std = None
    
    def prepare_data(self):
        """Скачивание данных (вызывается только на 1 GPU)"""
        pass
    
    def setup(self, stage=None):
        """Подготовка данных (вызывается на каждом GPU)"""
        # Создание трансформаций
        self.train_transform, self.val_transform, self.channel_mean, self.channel_std = create_transforms(self.cfg)
        
        # Создание датасета
        self.dataset = DogDataset(
            img_path=f"{self.cfg.dataset.paths.data_dir}/{self.cfg.dataset.paths.train_images}",
            csv_path=f"{self.cfg.dataset.paths.data_dir}/{self.cfg.dataset.paths.train_labels}",
            transform=self.train_transform
        )
        
        # Разделение на train/val
        indexes = list(range(len(self.dataset)))
        train_indexes, val_indexes = train_test_split(
            indexes,
            test_size=self.cfg.dataset.preprocessing.test_size,
            random_state=self.cfg.model.model.seed
        )
        
        self.train_dataset = Subset(self.dataset, train_indexes)
        self.val_dataset = Subset(self.dataset, val_indexes)
        
        # Применение разных трансформаций
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        
        print(f"Количество образцов в train_dataset: {len(self.train_dataset)}")
        print(f"Количество образцов в val_dataset: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        """DataLoader для обучения"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.train.batch_size,
            shuffle=self.cfg.train.train.shuffle,
            num_workers=self.cfg.train.train.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """DataLoader для валидации"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.train.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.train.train.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """DataLoader для тестирования (можно добавить тестовый датасет)"""
        return self.val_dataloader()


def get_device(cfg):
    """Определяет устройство для вычислений"""
    if cfg.train.train.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Используется CUDA")
        else:
            device = torch.device("cpu")
            print("Используется CPU")
    else:
        device = torch.device(cfg.train.train.device)
        print(f"Используется устройство из конфига: {device}")
    
    return device


def create_transforms(cfg):
    """Создает трансформации для изображений на основе конфигурации"""
    channel_mean = torch.Tensor(cfg.dataset.preprocessing.channel_mean)
    channel_std = torch.Tensor(cfg.dataset.preprocessing.channel_std)
    resize = cfg.dataset.preprocessing.resize
    crop = cfg.dataset.preprocessing.image_size
    
    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.RandomHorizontalFlip(p=cfg.dataset.preprocessing.random_horizontal_flip),
        transforms.RandomRotation(degrees=cfg.dataset.preprocessing.random_rotation),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ])
    
    return train_transform, valid_transform, channel_mean, channel_std