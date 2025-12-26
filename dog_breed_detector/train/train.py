import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
from ..dataset.dataset import DogDataset, TestDataset
from ..model.vit_model import PretrainViT
from utils import (
    get_device, create_transforms, split_dataset, 
    create_dataloaders, get_accuracy, download_data, show_samples
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Основная функция обучения модели"""
    
    # Установка seed и выбор устройства
    torch.manual_seed(cfg.model.model.seed)
    device = get_device(cfg.train.train.device)
    
    # Скачивание данных
    download_data(cfg)
    
    # Создание трансформаций
    train_transform, valid_transform, channel_mean, channel_std = create_transforms(cfg)
    
    # Создание датасета
    dataset = DogDataset(
        img_path=f"{cfg.dataset.paths.data_dir}/{cfg.dataset.paths.train_images}",
        csv_path=f"{cfg.dataset.paths.data_dir}/{cfg.dataset.paths.train_labels}",
        transform=train_transform
    )
    
    # Разделение на train/valid
    train_dataset, valid_dataset = split_dataset(dataset, test_size=cfg.model.preprocessing.test_size)
    
    # Применение разных трансформаций для train и valid
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = valid_transform
    
    # Создание DataLoader
    train_dataloader, valid_dataloader = create_dataloaders(train_dataset, valid_dataset, cfg)
    
    # Визуализация примеров из батча
    if cfg.train.train.visualize_samples:
        batch_img, batch_label = next(iter(train_dataloader))
        fig = show_samples(
            batch_img, 
            batch_label, 
            num_samples=cfg.train.train.batch_size,
            label_idx2name=dataset.label_idx2name,
            channel_mean=channel_mean,
            channel_std=channel_std,
            crop_size=cfg.model.preprocessing.image_size
        )
        plt.savefig("samples.png")
        plt.close(fig)
    
    # Создание модели
    net = PretrainViT(cfg)
    net.to(device)
    print(f"Количество обучаемых параметров: {sum([param.numel() for param in net.parameters() if param.requires_grad])}")
    
    # Loss функция и Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), 
        lr=cfg.model.optimizer.learning_rate, 
        momentum=cfg.model.optimizer.momentum
    )
    
    # Обучение модели
    train_model(net, train_dataloader, valid_dataloader, criterion, optimizer, device, cfg)
    
    # Предсказания на тестовых данных
    # make_predictions(net, dataset, valid_transform, device, cfg)


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, cfg):
    """Обучает модель и возвращает историю loss"""
    valid_loss_history = []
    
    # Цикл обучения по эпохам
    for epoch in range(cfg.train.train.epochs):
        # Обучение на одной эпохе
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, device, cfg
        )
        
        # Валидация
        valid_loss, valid_acc = validate(
            model, valid_dataloader, criterion, device
        )
        
        # Вывод метрик
        print(f"Эпоха: {epoch:2d}, "
              f"train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, "
              f"valid loss: {valid_loss:.3f}, valid acc: {valid_acc:.3f}")
        
        # Сохранение истории метрики валидации
        valid_loss_history.append(valid_loss)
        
        # Сохранение лучшей модели
        if valid_loss <= min(valid_loss_history):
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Модель сохранена как лучшая (valid loss: {valid_loss:.3f})")
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), "final_model.pt")
    
    return valid_loss_history


def train_epoch(model, dataloader, criterion, optimizer, device, cfg):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    running_acc = 0.0
    total_acc = 0.0
    
    # Количество батчей для усреднения лосса
    log_interval = cfg.train.train.log_interval
    
    for batch_idx, (batch_img, batch_label) in enumerate(dataloader):
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_img)
        loss = criterion(output, batch_label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Сбор статистики
        running_loss += loss.item()
        total_loss += loss.item()
        
        acc = get_accuracy(output, batch_label)
        running_acc += acc
        total_acc += acc
        
        # Логирование каждые log_interval батчей
        if batch_idx % log_interval == 0 and batch_idx != 0:
            avg_loss = running_loss / log_interval
            avg_acc = running_acc / log_interval
            print(f"[шаг: {batch_idx:4d}/{len(dataloader)}] loss: {avg_loss:.3f}, acc: {avg_acc:.3f}")
            running_loss = 0.0
            running_acc = 0.0
    
    # Возвращаем средние метрики за эпоху
    return total_loss / len(dataloader), total_acc / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for batch_img, batch_label in dataloader:
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            
            output = model(batch_img)
            loss = criterion(output, batch_label)
            total_loss += loss.item()
            
            acc = get_accuracy(output, batch_label)
            total_acc += acc
    
    return total_loss / len(dataloader), total_acc / len(dataloader)



# ПЕРЕДЕЛАТЬ, НУЖНО ТОЛЬКО ДЛЯ kaggle
# def make_predictions(model, train_dataset, transform_fn, device, cfg):
#     """Делает предсказания на тестовых данных и сохраняет submission файл"""
#     print("Начало предсказаний на тестовых данных...")
    
#     # Загрузка модели с лучшими весами
#     model.load_state_dict(torch.load("best_model.pt", map_location=device))
#     model.to(device)
#     model.eval()
    
#     # Загрузка данных для submission
#     submit_df = pd.read_csv(f"{cfg.dataset.paths.data_dir}/{cfg.dataset.paths.sample_submission}")
#     test_names = submit_df["id"].values
#     columns = list(train_dataset.label_idx2name)
    
#     dfs = []
    
#     # Предсказания с прогресс-баром
#     with torch.no_grad():
#         for batch_img, batch_name in tqdm(test_dataloader, desc="Предсказание"):
#             df = pd.DataFrame(columns=["id"] + columns)
#             df["id"] = batch_name
            
#             batch_img = batch_img.to(device)
#             output = model(batch_img)
#             sm = torch.nn.functional.softmax(output, dim=1)
#             df[columns] = sm.cpu().numpy()
#             dfs.append(df)
    
#     # Объединение всех предсказаний
#     my_submit = pd.concat(dfs, ignore_index=True)
#     my_submit.to_csv("submit.csv", index=False)
#     print(f"Submission файл сохранен как submit.csv, количество предсказаний: {len(my_submit)}")


if __name__ == "__main__":
    main()