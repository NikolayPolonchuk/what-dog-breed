import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def get_accuracy(output, label):
    """Вычисляет точность предсказаний"""
    output = output.to("cpu")
    label = label.to("cpu")
    
    sm = F.softmax(output, dim=1)
    _, index = torch.max(sm, dim=1)
    return torch.sum((label == index)) / label.size()[0]


def show_samples(batch_img, batch_label, num_samples, label_idx2name, channel_mean, channel_std, crop_size):
    """Показывает примеры изображений с метками"""
    sample_idx = 0
    total_col = 4
    total_row = math.ceil(num_samples / total_col)
    col_idx = 0
    row_idx = 0
    
    fig, axs = plt.subplots(total_row, total_col, figsize=(15, 15))
    
    while sample_idx < num_samples:
        img = batch_img[sample_idx]
        
        # Денормализация изображения
        img = img.view(3, -1) * channel_std.view(3, -1) + channel_mean.view(3, -1)
        img = img.view(3, crop_size, crop_size)
        img = img.permute(1, 2, 0)
        img = torch.clamp(img, 0, 1)  # Ограничиваем значения
        
        axs[row_idx, col_idx].imshow(img)
        
        if batch_label is not None and label_idx2name is not None:
            axs[row_idx, col_idx].set_title(label_idx2name[batch_label[sample_idx]])
        
        sample_idx += 1
        col_idx += 1
        if col_idx == total_col:
            col_idx = 0
            row_idx += 1
    
    plt.tight_layout()
    return fig


def plot_training_history(history, current_epoch):
    """Исправленная версия с проверкой размеров данных"""
    
    # Проверяем, что есть хотя бы одна эпоха данных
    if len(history['train_loss']) == 0:
        print(f"История пуста, невозможно построить график для эпохи {current_epoch}")
        return None
    
    # Создаем список эпох на основе доступных данных
    # Используем длину train_loss как базовую
    available_epochs = len(history['train_loss'])
    epochs_list = list(range(1, available_epochs + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График loss
    if len(history['train_loss']) >= available_epochs and len(history['valid_loss']) >= available_epochs:
        axes[0].plot(epochs_list, history['train_loss'][:available_epochs], 'b-', label='Train Loss', marker='o')
        axes[0].plot(epochs_list, history['valid_loss'][:available_epochs], 'r-', label='Valid Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Loss History (Эпоха {current_epoch})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        print(f"Недостаточно данных для построения графика loss. Train: {len(history['train_loss'])}, Valid: {len(history['valid_loss'])}")
    
    # График accuracy
    train_acc_len = len(history.get('train_accuracy', []))
    valid_acc_len = len(history.get('valid_accuracy', []))
    
    if train_acc_len > 0 and valid_acc_len > 0:
        min_acc_epochs = min(train_acc_len, valid_acc_len)
        axes[1].plot(
            epochs_list[:min_acc_epochs], 
            history['train_accuracy'][:min_acc_epochs], 
            'b-', label='Train Accuracy', marker='o'
        )
        axes[1].plot(
            epochs_list[:min_acc_epochs], 
            history['valid_accuracy'][:min_acc_epochs], 
            'r-', label='Valid Accuracy', marker='s'
        )
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Accuracy History (Эпоха {current_epoch})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        print(f"Недостаточно данных для построения графика accuracy. Train: {train_acc_len}, Valid: {valid_acc_len}")
        # Если нет accuracy, показываем только один график
        fig.delaxes(axes[1])
        fig.set_size_inches(6, 4)
    
    plt.tight_layout()
    return fig


def calculate_f1_score(model, dataloader, device):
    """Вычисление F1-score"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_img, batch_label in dataloader:
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_img)
            predictions = torch.argmax(output, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())
    
    # Проверка, что есть данные для расчета
    if len(all_preds) == 0:
        return 0.0
    
    return f1_score(all_labels, all_preds, average='weighted')