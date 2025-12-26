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


def plot_training_history(history, current_epoch, total_epochs):
    """Построение графиков после каждой эпохи"""
    epochs_list = list(range(1, current_epoch + 1))
    
    # Создание subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # График Accuracy
    if len(history['train_accuracy']) > 0:
        axes[0].plot(epochs_list, history['train_accuracy'], 'b-', label='Train Accuracy', marker='o')
        axes[0].plot(epochs_list, history['valid_accuracy'], 'r-', label='Valid Accuracy', marker='s')
    axes[0].set_title(f'Model Accuracy (Epoch {current_epoch}/{total_epochs})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs_list)
    
    # График F1-Score
    if len(history['train_f1']) > 0:
        axes[1].plot(epochs_list, history['train_f1'], 'b-', label='Train F1-Score', marker='o')
        axes[1].plot(epochs_list, history['valid_f1'], 'r-', label='Valid F1-Score', marker='s')
    axes[1].set_title(f'Model F1-Score (Epoch {current_epoch}/{total_epochs})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1-Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs_list)
    
    # График Loss
    if len(history['train_loss']) > 0:
        axes[2].plot(epochs_list, history['train_loss'], 'b-', label='Train Loss', marker='o')
        axes[2].plot(epochs_list, history['valid_loss'], 'r-', label='Valid Loss', marker='s')
    axes[2].set_title(f'Model Loss (Epoch {current_epoch}/{total_epochs})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(epochs_list)
    
    plt.tight_layout()
    plt.show()


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