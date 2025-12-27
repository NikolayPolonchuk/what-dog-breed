import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import os
from pathlib import Path
from hydra.utils import get_original_cwd

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

# ВАЖНО: правильные относительные импорты
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Теперь можно импортировать
try:
    from dataset.dataset import DogDataModule
    from model.vit_model import PretrainViT
    from callbacks.mlflow_callback import MLFlowCallback
    from dataset.download_data import download_data
    from train.utils import get_accuracy, show_samples, plot_training_history
except ImportError:
    # Альтернативный путь
    from ..dataset.dataset import DogDataModule
    from ..model.vit_model import PretrainViT
    from ..callbacks.mlflow_callback import MLFlowCallback
    from ..dataset.download_data import download_data
    from ..train.utils import get_accuracy, show_samples, plot_training_history


class LitDogModel(pl.LightningModule):
    """LightningModule для классификации собак"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Инициализация модели
        self.model = PretrainViT(cfg)
        self.criterion = nn.CrossEntropyLoss()
        
        # История метрик (начинаем с пустых списков)
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
        
        # Логирование с on_epoch=True для корректного агрегирования
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Вызывается в конце каждой эпохи обучения"""
        # Собираем метрики за эпоху из логгера
        train_loss_epoch = self.trainer.callback_metrics.get('train_loss', None)
        train_acc_epoch = self.trainer.callback_metrics.get('train_acc', None)
        
        if train_loss_epoch is not None:
            self.history['train_loss'].append(float(train_loss_epoch))
        if train_acc_epoch is not None:
            self.history['train_accuracy'].append(float(train_acc_epoch))
    
    def validation_step(self, batch, batch_idx):
        """Шаг валидации"""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Вычисление метрик
        acc = get_accuracy(y_hat, y)
        
        # Логирование валидационных метрик
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        """Вызывается в конце каждой эпохи валидации"""
        # Собираем валидационные метрики
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        val_acc = self.trainer.callback_metrics.get('val_acc', None)
        
        if val_loss is not None:
            self.history['valid_loss'].append(float(val_loss))
        if val_acc is not None:
            self.history['valid_accuracy'].append(float(val_acc))
        
        # Обновление графиков каждые N эпох
        if (self.current_epoch + 1) % self.cfg.train.train.get('plot_interval', 10) == 0:
            # Создаем график
            fig = plot_training_history(
                self.history,
                self.current_epoch + 1
            )
            
            # Сохраняем график локально
            if fig is not None:
                plot_dir = Path("training_plots")
                plot_dir.mkdir(exist_ok=True)
                plot_path = plot_dir / f"history_epoch_{self.current_epoch+1}.png"
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"График сохранен: {plot_path}")
    
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
    
    def _calculate_f1_batch(self, predictions, targets):
        """Вычисление F1-score для батча"""
        preds = torch.argmax(predictions, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Основная функция обучения с PyTorch Lightning и MLFlow"""
    
    # Установка seed
    pl.seed_everything(cfg.model.model.seed, workers=True)
    
    # Скачивание данных
    download_data(cfg)
    
    # Создание DataModule
    data_module = DogDataModule(cfg)
    
    # Инициализация модели
    model = LitDogModel(cfg)
    
    # Создаем директорию для чекпоинтов
    checkpoint_dir = Path(cfg.train.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Callback'и
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss' if cfg.model.model.criterion == 'val_loss' else 'val_acc',
        dirpath=str(checkpoint_dir),
        filename='dog-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=cfg.train.train.get('save_top_k', 3),
        mode='min' if cfg.model.model.criterion == 'val_loss' else 'max',
        save_last=True,
        save_on_train_epoch_end=False,  # Сохраняем после валидации
        every_n_epochs=1,  # Сохраняем каждую эпоху
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor=cfg.model.model.criterion,
        patience=cfg.train.train.early_stopping_patience,
        mode='min' if cfg.model.model.criterion == 'val_loss' else 'max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # MLFlow логгер и callback
    mlflow_logger = None
    mlflow_callback = None
    
    if cfg.mlflow.mlflow.enabled:
        # Создаем MLFlow логгер
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.mlflow.tracking_uri,
            run_name=cfg.mlflow.mlflow.run_name,
            tags={
                "project": cfg.dataset.paths.name,
                "model": cfg.model.model.name,
                "dataset": cfg.dataset.paths.name
            }
        )
        
        # Создаем mlflow callback
        mlflow_callback = MLFlowCallback(cfg)
    
    # Callbacks list
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    if mlflow_callback:
        callbacks.append(mlflow_callback)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.train.epochs,
        accelerator='auto',
        devices='auto',
        logger=mlflow_logger if mlflow_logger else True,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.train.log_interval,
        enable_progress_bar=True,
        deterministic=True,
        gradient_clip_val=cfg.train.train.gradient_clip_val,
        enable_checkpointing=True,  # Включаем систему чекпоинтов
        default_root_dir=".",  # Базовая директория для логов
    )
    
    # Визуализация примеров с логированием в mlflow
    if cfg.train.train.visualize_samples:
        data_module.setup('fit')
        train_loader = data_module.train_dataloader()
        batch_img, batch_label = next(iter(train_loader))
        
        fig = show_samples(
            batch_img,
            batch_label,
            num_samples=min(8, cfg.train.train.batch_size),
            label_idx2name=data_module.dataset.label_idx2name,
            channel_mean=data_module.channel_mean,
            channel_std=data_module.channel_std,
            crop_size=cfg.dataset.preprocessing.image_size
        )
        
        # Сохраняем локально
        plt.savefig("samples.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Логируем в mlflow через callback
        if mlflow_callback:
            mlflow_callback.log_samples(
                batch_img, batch_label,
                data_module.dataset.label_idx2name,
                data_module.channel_mean,
                data_module.channel_std,
                cfg.dataset.preprocessing.image_size
            )
    
    try:
        # Обучение
        trainer.fit(model, datamodule=data_module)
        
    except Exception as e:
        print(f"Обучение прервано с ошибкой: {e}")
        print("Сохраняем текущее состояние модели...")
        
        # Сохраняем модель при ошибке
        torch.save(model.state_dict(), checkpoint_dir / "interrupted_model.pt")
        print(f"Модель сохранена в {checkpoint_dir}/interrupted_model.pt")
        
        # Сохраняем чекпоинт вручную
        trainer.save_checkpoint(checkpoint_dir / "last_interrupted.ckpt")
        raise
    
    # Тестирование на лучшей модели
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"\nЗагрузка лучшей модели: {best_model_path}")
        model = LitDogModel.load_from_checkpoint(best_model_path, cfg=cfg)
        trainer.test(model, datamodule=data_module)
    else:
        print("\nЛучшая модель не найдена, используем текущую модель")
    
    # Финальные графики
    print("\nФИНАЛЬНЫЕ ГРАФИКИ ОБУЧЕНИЯ")
    if len(model.history['train_loss']) > 0:
        fig = plot_training_history(
            model.history,
            trainer.current_epoch + 1
        )
        
        # Сохраняем финальный график
        if fig:
            plt.savefig("training_history_final.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Логируем в mlflow
            if mlflow_logger and hasattr(mlflow_logger.experiment, 'log_artifact'):
                mlflow_logger.experiment.log_artifact(
                    mlflow_logger.run_id,
                    "training_history_final.png",
                    "plots"
                )
    
    # Сохранение финальной модели
    final_model_path = "final_model.pt"
    torch.save({
        'epoch': trainer.current_epoch,
        'model_state_dict': model.state_dict(),
        'history': model.history,
        'cfg': cfg
    }, final_model_path)
    print(f"Финальная модель сохранена: {final_model_path}")
    
    # Логируем финальную модель в mlflow
    if mlflow_logger and hasattr(mlflow_logger.experiment, 'log_artifact'):
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            final_model_path,
            "models"
        )
    
    print(f"\nОбучение завершено!")
    print(f"Чекпоинты сохранены в: {checkpoint_dir}")
    if cfg.mlflow.mlflow.enabled and mlflow_logger:
        print(f"MLFlow эксперимент: {cfg.mlflow.mlflow.experiment_name}")
        print(f"MLFlow run ID: {mlflow_logger.run_id}")
    
    # Выводим список созданных чекпоинтов
    print("\nСозданные чекпоинты:")
    for checkpoint_file in checkpoint_dir.glob("*.ckpt"):
        print(f"  - {checkpoint_file}")


if __name__ == "__main__":
    main()