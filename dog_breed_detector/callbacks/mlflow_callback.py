import mlflow
import mlflow.pytorch
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
import tempfile
import os
import time
import math
from typing import List, Optional


class MLFlowCallback(Callback):
    """MLFlow callback для PyTorch Lightning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlflow_config = config.mlflow.mlflow
        self.enabled = self.mlflow_config.enabled
        self.experiment_id = None
        self.run_id = None
        self._temp_files: List[str] = []
        
        if self.enabled:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Настройка MLFlow"""
        tracking_uri = self.mlflow_config.tracking_uri
        experiment_name = self.mlflow_config.experiment_name
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Очистка перед настройкой
        self._cleanup_active_runs()
        
        # Создание или получение эксперимента
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Создан новый эксперимент: {experiment_name} (ID: {self.experiment_id})")
        else:
            self.experiment_id = experiment.experiment_id
            print(f"Используем существующий эксперимент: {experiment_name} (ID: {self.experiment_id})")
        
        print(f"MLFlow настроен:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment: {experiment_name}")

    def _cleanup_active_runs(self):
        """Очистка активных MLFlow runs"""
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                print(f"Найден активный MLFlow run: {active_run.info.run_id}. Завершаем...")
                mlflow.end_run()
                print("Активный run завершен")
            else:
                print("Нет активных MLFlow runs")
        except Exception as e:
            print(f"Ошибка при очистке активных runs: {e}")
    
    def on_fit_start(self, trainer, pl_module):
        """Начало обучения"""
        if not self.enabled:
            return
        
        try:
            # Гарантируем чистый старт
            self._cleanup_active_runs()
            
            # Проверяем еще раз
            if mlflow.active_run() is not None:
                print("Предупреждение: обнаружен активный run, завершаем явно...")
                mlflow.end_run()
            
            # Старт run
            run_name = self.mlflow_config.run_name
            mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=False
            )
            self.run_id = mlflow.active_run().info.run_id
            
            # Логирование параметров
            if self.mlflow_config.logging.params:
                self._log_params(pl_module)
            
            print(f"MLFlow run начат: {run_name} (ID: {self.run_id})")
            
        except Exception as e:
            print(f"Ошибка при старте MLFlow: {e}")
            self.enabled = False
    
    def _log_params(self, pl_module):
        """Логирование параметров"""
        try:
            # Логирование конфигурации
            config_dict = OmegaConf.to_container(self.config.train, resolve=True)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                config_path = f.name
                self._temp_files.append(config_path)
            
            mlflow.log_artifact(config_path, "config")
            
            # Логирование гиперпараметров модели
            if hasattr(pl_module, 'hparams'):
                params = {}
                for key, value in pl_module.hparams.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        params[key] = value
                    elif isinstance(value, (list, tuple, dict)):
                        params[key] = str(value)
                
                mlflow.log_params(params)
            
            print(f"Параметры залогированы в MLFlow")
            
        except Exception as e:
            print(f"Ошибка при логировании параметров: {e}")
    
    def _save_figure(self, fig: plt.Figure, artifact_path: str, filename_prefix: str = "plot") -> Optional[str]:
        """Сохранение и логирование графиков"""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', prefix=filename_prefix, delete=False) as tmp:
                temp_file = tmp.name
                fig.savefig(temp_file, dpi=150, bbox_inches='tight')
            
            mlflow.log_artifact(temp_file, artifact_path)
            self._temp_files.append(temp_file)
            return temp_file
            
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")
            if temp_file and os.path.exists(temp_file):
                self._safe_delete(temp_file)
            return None
    
    def _safe_delete(self, filepath: str, max_attempts: int = 3):
        """Удаление файлов"""
        if not filepath or not os.path.exists(filepath):
            return
        
        for attempt in range(max_attempts):
            try:
                os.unlink(filepath)
                if filepath in self._temp_files:
                    self._temp_files.remove(filepath)
                break
                
            except (PermissionError, OSError) as e:
                if attempt == max_attempts - 1:
                    print(f"Не удалось удалить файл {filepath} после {max_attempts} попыток: {e}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Ошибка при удалении {filepath}: {e}")
                break
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Конец эпохи обучения"""
        if not self.enabled:
            return
        
        # Логирование метрик
        if self.mlflow_config.logging.metrics:
            try:
                metrics = trainer.callback_metrics
                epoch_metrics = {}
                
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key] = value.item()
                    else:
                        epoch_metrics[key] = value
                
                epoch_metrics['epoch'] = trainer.current_epoch
                mlflow.log_metrics(epoch_metrics, step=trainer.current_epoch)
                
            except Exception as e:
                print(f"Ошибка при логировании метрик: {e}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Конец эпохи валидации"""
        if not self.enabled:
            return
        
        # Логирование графиков раз в 5 эпох
        if (self.mlflow_config.logging.plots and 
            hasattr(pl_module, 'history') and
            trainer.current_epoch % 5 == 0):
            
            try:
                history = pl_module.history
                
                # Проверяем наличие данных
                train_loss = history.get('train_loss', [])
                valid_loss = history.get('valid_loss', [])
                
                if len(train_loss) == 0 or len(valid_loss) == 0:
                    return
                
                # Определяем количество доступных эпох
                available_epochs = min(len(train_loss), len(valid_loss))
                epochs = list(range(1, available_epochs + 1))
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # График loss
                axes[0].plot(epochs, train_loss[:available_epochs], 'b-', label='Train Loss', linewidth=2)
                axes[0].plot(epochs, valid_loss[:available_epochs], 'r-', label='Valid Loss', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title(f'Loss History (Epoch {trainer.current_epoch})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # График accuracy
                train_acc = history.get('train_accuracy', [])
                valid_acc = history.get('valid_accuracy', [])
                
                if len(train_acc) > 0 and len(valid_acc) > 0:
                    acc_epochs = min(len(train_acc), len(valid_acc))
                    axes[1].plot(epochs[:acc_epochs], train_acc[:acc_epochs], 'b-', label='Train Accuracy', linewidth=2)
                    axes[1].plot(epochs[:acc_epochs], valid_acc[:acc_epochs], 'r-', label='Valid Accuracy', linewidth=2)
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Accuracy')
                    axes[1].set_title(f'Accuracy History (Epoch {trainer.current_epoch})')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                else:
                    fig.delaxes(axes[1])
                    fig.set_size_inches(6, 4)
                
                plt.tight_layout()
                
                # Сохраняем и логируем график
                self._save_figure(fig, "plots", f"training_history_epoch_{trainer.current_epoch}")
                plt.close(fig)
                    
            except Exception as e:
                print(f"Ошибка при создании графиков: {e}")
    
    def on_train_end(self, trainer, pl_module):
        """Конец обучения"""
        if not self.enabled:
            return
        
        try:
            # Логирование модели
            if self.mlflow_config.logging.models:
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=pl_module,
                        artifact_path="model",
                        registered_model_name=f"dog-breed-vit-{self.run_id}"
                    )
                    print(f"Модель залогирована в MLFlow")
                except Exception as e:
                    print(f"Ошибка при логировании модели: {e}")
            
            # Логирование чекпоинтов
            if (self.mlflow_config.artifacts.save_checkpoints and
                hasattr(trainer, 'checkpoint_callback') and
                trainer.checkpoint_callback is not None):
                
                best_model_path = trainer.checkpoint_callback.best_model_path
                if best_model_path and os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "checkpoints")
                    print(f"Лучший чекпоинт залогирован: {best_model_path}")
            
            # Финальные метрики
            final_metrics = {}
            if hasattr(pl_module, 'history'):
                history = pl_module.history
                
                if history.get('train_loss'):
                    final_metrics['final_train_loss'] = history['train_loss'][-1]
                if history.get('valid_loss'):
                    final_metrics['final_val_loss'] = history['valid_loss'][-1]
                if history.get('train_accuracy'):
                    final_metrics['final_train_acc'] = history['train_accuracy'][-1]
                if history.get('valid_accuracy'):
                    final_metrics['final_val_acc'] = history['valid_accuracy'][-1]
                
                final_metrics['total_epochs'] = trainer.current_epoch + 1
            
            if final_metrics:
                mlflow.log_metrics(final_metrics)
                print(f"Финальные метрики: {final_metrics}")
            
            # Завершение run
            mlflow.end_run()
            print(f"MLFlow run завершен (ID: {self.run_id})")
            
        finally:
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        files_to_delete = list(self._temp_files)
        for filepath in files_to_delete:
            self._safe_delete(filepath)
        
        print(f"Очищено {len(files_to_delete) - len(self._temp_files)} временных файлов")
    
    def log_samples(self, batch_img, batch_label, label_idx2name, channel_mean, channel_std, crop_size):
        """Логирование примеров изображений"""
        if not self.enabled:
            return
        
        if not self.mlflow_config.artifacts.save_samples:
            return
        
        try:
            # Используем локальную функцию
            fig = self._create_samples_figure(
                batch_img, batch_label, 
                min(8, len(batch_img)),
                label_idx2name,
                channel_mean,
                channel_std,
                crop_size
            )
            
            if fig:
                saved_path = self._save_figure(fig, "samples", "data_samples")
                if saved_path:
                    print(f"Примеры изображений залогированы в MLFlow")
                plt.close(fig)
                
        except Exception as e:
            print(f"Ошибка при логировании примеров: {e}")
    
    def _create_samples_figure(self, batch_img, batch_label, num_samples, label_idx2name, channel_mean, channel_std, crop_size):
        """Создает график с примерами"""
        sample_idx = 0
        total_col = 4
        total_row = math.ceil(num_samples / total_col)
        col_idx = 0
        row_idx = 0
        
        fig, axs = plt.subplots(total_row, total_col, figsize=(15, 15))
        
        while sample_idx < num_samples:
            img = batch_img[sample_idx]
            
            # Денормализация
            img = img.view(3, -1) * channel_std.view(3, -1) + channel_mean.view(3, -1)
            img = img.view(3, crop_size, crop_size)
            img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            
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
    
    def __del__(self):
        """Деструктор"""
        if hasattr(self, '_temp_files') and self._temp_files:
            print(f"MLFlowCallback: очистка {len(self._temp_files)} временных файлов в деструкторе")
            self._cleanup_temp_files()