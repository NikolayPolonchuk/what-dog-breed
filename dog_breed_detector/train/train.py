import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.dataset import DogDataModule
from model.vit_model import PretrainViT
from dataset.download_data import download_data
from .utils import (
    get_accuracy, show_samples, plot_training_history
)


class LitDogModel(pl.LightningModule):
    """LightningModule для классификации собак"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Инициализация модели
        self.model = PretrainViT(cfg)
        self.criterion = nn.CrossEntropyLoss()
        
        # История метрик
        self.history = {
            'train_loss': [], 'valid_loss': [],
            'train_accuracy': [], 'valid_accuracy': [],
            'train_f1': [], 'valid_f1': []
        }
    
    def forward(self, x):
        """Инференс модели"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Вычисление метрик
        acc = get_accuracy(y_hat, y)
        train_f1 = self._calculate_f1_batch(y_hat, y)
        
        # Логирование
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', train_f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Шаг валидации"""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Вычисление метрик
        acc = get_accuracy(y_hat, y)
        val_f1 = self._calculate_f1_batch(y_hat, y)
        
        # Логирование
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', val_f1)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        """Шаг тестирования"""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = get_accuracy(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        """Настройка оптимизаторов и шедулеров"""
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.cfg.model.optimizer.learning_rate,
            momentum=self.cfg.model.optimizer.momentum
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.model.optimizer.step_size,
            gamma=self.cfg.model.optimizer.gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Вызывается в конце каждой эпохи обучения"""
        # Сбор метрик
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        train_acc = self.trainer.callback_metrics.get('train_acc', torch.tensor(0.0))
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        val_acc = self.trainer.callback_metrics.get('val_acc', torch.tensor(0.0))
        
        # Сохранение истории
        self.history['train_loss'].append(train_loss.item())
        self.history['valid_loss'].append(val_loss.item())
        self.history['train_accuracy'].append(train_acc.item())
        self.history['valid_accuracy'].append(val_acc.item())
    
    def on_validation_epoch_end(self):
        """Вызывается в конце каждой эпохи валидации"""
        # Обновление графиков каждые N эпох
        if (self.current_epoch + 1) % self.cfg.train.train.plot_interval == 0:
            plot_training_history(
                self.history,
                self.current_epoch + 1,
                self.cfg.train.train.epochs
            )
    
    def _calculate_f1_batch(self, predictions, targets):
        """Вычисление F1-score для батча"""
        preds = torch.argmax(predictions, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Основная функция обучения с PyTorch Lightning"""
    
    # Установка seed
    pl.seed_everything(cfg.model.model.seed, workers=True)
    
    # Скачивание данных
    download_data(cfg)
    
    # Создание DataModule
    data_module = DogDataModule(cfg)
    
    # Инициализация модели
    model = LitDogModel(cfg)
    
    # Callback'и
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.model.criterion,
        dirpath='../plots/checkpoints/',
        filename='dog-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=cfg.model.model.criterion,
        patience=cfg.train.train.early_stopping_patience,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Логгер
    logger = TensorBoardLogger("tb_logs", name="dog_classification")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.train.epochs,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=cfg.train.train.log_interval,
        enable_progress_bar=True,
        deterministic=True,
        gradient_clip_val=cfg.train.train.gradient_clip_val
    )
    
    # Визуализация примеров
    if cfg.train.train.visualize_samples:
        data_module.setup('fit')
        train_loader = data_module.train_dataloader()
        batch_img, batch_label = next(iter(train_loader))
        
        fig = show_samples(
            batch_img,
            batch_label,
            num_samples=cfg.train.train.batch_size,
            label_idx2name=data_module.dataset.label_idx2name,
            channel_mean=data_module.channel_mean,
            channel_std=data_module.channel_std,
            crop_size=cfg.dataset.preprocessing.image_size
        )
        plt.savefig("samples.png")
        plt.close(fig)
    
    # Обучение
    trainer.fit(model, datamodule=data_module)
    
    # Тестирование на лучшей модели
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nЗагрузка лучшей модели: {best_model_path}")
        model = LitDogModel.load_from_checkpoint(best_model_path, cfg=cfg)
        trainer.test(model, datamodule=data_module)
    
    # Финальные графики
    print("\nФИНАЛЬНЫЕ ГРАФИКИ ОБУЧЕНИЯ")
    plot_training_history(
        model.history,
        cfg.train.train.epochs,
        cfg.train.train.epochs
    )
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), "final_model.pt")
    print("Финальная модель сохранена: final_model.pt")


if __name__ == "__main__":
    main()