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
        self._temp_files: List[str] = []  # Для отслеживания временных файлов
        
        if self.enabled:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Настройка MLFlow"""
        tracking_uri = self.mlflow_config.tracking_uri
        experiment_name = self.mlflow_config.experiment_name
        
        mlflow.set_tracking_uri(tracking_uri)

        # Проверяем активные runs и завершаем их если есть
        self._cleanup_active_runs()
        
        # Создание или получение эксперимента
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        print(f"MLFlow настроен:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment: {experiment_name} (ID: {self.experiment_id})")

    def _cleanup_active_runs(self):
        """Очистка активных MLFlow runs"""
        try:
            # Безусловное завершение любого активного run
            mlflow.end_run()
            print("Проведена очистка активных MLFlow runs")
        except Exception as e:
            # Если не было активного run, mlflow.end_run() может вызвать исключение
            print(f"Нет активного run для завершения или ошибка: {e}")
    
    def on_fit_start(self, trainer, pl_module):
        """Начало обучения"""
        if not self.enabled:
            return
        
        try:
            # очищаем
            self._cleanup_active_runs()
            # Старт run
            run_name = self.mlflow_config.run_name
            mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name
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
            
            # Создаем временный файл для конфига
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                config_path = f.name
                self._temp_files.append(config_path)  # Добавляем в список для очистки
            
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
            print(f"Ошибка при логировании параметров в MLFlow: {e}")
    
    def _save_figure(self, fig: plt.Figure, artifact_path: str, filename_prefix: str = "plot") -> Optional[str]:
        """Сохранение и логирование графиков"""
        temp_file = None
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(suffix='.png', prefix=filename_prefix, delete=False) as tmp:
                temp_file = tmp.name
                fig.savefig(temp_file, dpi=150, bbox_inches='tight')
            
            # Логируем артефакт
            mlflow.log_artifact(temp_file, artifact_path)
            
            # Добавляем в список для очистки
            self._temp_files.append(temp_file)
            
            return temp_file
            
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")
            if temp_file and os.path.exists(temp_file):
                self._safe_delete(temp_file)
            return None
    
    def _safe_delete(self, filepath: str, max_attempts: int = 3):
        """Удаление старых файлов"""
        if not filepath or not os.path.exists(filepath):
            return
        
        for attempt in range(max_attempts):
            try:
                os.unlink(filepath)
                # Удаляем из списка если удалось
                if filepath in self._temp_files:
                    self._temp_files.remove(filepath)
                break  # Успешно удалили
                
            except (PermissionError, OSError) as e:
                if attempt == max_attempts - 1:  # Последняя попытка
                    print(f"Не удалось удалить файл {filepath} после {max_attempts} попыток: {e}")
                else:
                    time.sleep(0.1)  # Ждем 100ms перед повторной попыткой
            except Exception as e:
                print(f"Неожиданная ошибка при удалении {filepath}: {e}")
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
                
                # Добавляем номер эпохи
                epoch_metrics['epoch'] = trainer.current_epoch
                
                mlflow.log_metrics(epoch_metrics, step=trainer.current_epoch)
                
            except Exception as e:
                print(f"Ошибка при логировании метрик: {e}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Конец эпохи валидации - логирование графиков"""
        if not self.enabled:
            return
        
        # Логирование графиков раз в 5 эпох
        if (self.mlflow_config.logging.plots and 
            hasattr(pl_module, 'history') and
            trainer.current_epoch % 5 == 0):
            
            try:
                # Создаем график истории обучения
                history = pl_module.history
                if len(history['train_loss']) > 0:
                    epochs = list(range(1, len(history['train_loss']) + 1))
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # График loss
                    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
                    axes[0].plot(epochs, history['valid_loss'], 'r-', label='Valid Loss', linewidth=2)
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].set_title(f'Loss History (Epoch {trainer.current_epoch})')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # График accuracy
                    if 'train_accuracy' in history and 'valid_accuracy' in history:
                        axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
                        axes[1].plot(epochs, history['valid_accuracy'], 'r-', label='Valid Accuracy', linewidth=2)
                        axes[1].set_xlabel('Epoch')
                        axes[1].set_ylabel('Accuracy')
                        axes[1].set_title(f'Accuracy History (Epoch {trainer.current_epoch})')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Сохраняем и логируем график
                    self._save_figure(fig, "plots", f"training_history_epoch_{trainer.current_epoch}")
                    
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Ошибка при создании графиков обучения: {e}")
    
    def on_train_end(self, trainer, pl_module):
        """Конец обучения - финальное логирование и очистка"""
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
                    print(f"Ошибка при логировании модели в MLFlow: {e}")
            
            # Логирование чекпоинтов
            if (self.mlflow_config.artifacts.save_checkpoints and
                hasattr(trainer, 'checkpoint_callback') and
                trainer.checkpoint_callback is not None):
                
                best_model_path = trainer.checkpoint_callback.best_model_path
                if best_model_path and os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "checkpoints")
            
            # Финальные метрики
            if hasattr(pl_module, 'history'):
                history = pl_module.history
                final_metrics = {
                    'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                    'final_val_loss': history['valid_loss'][-1] if history['valid_loss'] else 0,
                    'final_train_acc': history['train_accuracy'][-1] if history['train_accuracy'] else 0,
                    'final_val_acc': history['valid_accuracy'][-1] if history['valid_accuracy'] else 0,
                    'total_epochs': trainer.current_epoch + 1
                }
                mlflow.log_metrics(final_metrics)
            
            # Завершение run
            mlflow.end_run()
            print(f"MLFlow run завершен (ID: {self.run_id})")
            
        finally:
            # Всегда очищаем временные файлы
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Очистка всех временных файлов"""
        files_to_delete = list(self._temp_files)  # Копируем список
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
            from ..train.utils import show_samples
            
            # Создаем график с примерами
            fig = show_samples(
                batch_img, batch_label, 
                num_samples=min(8, len(batch_img)),
                label_idx2name=label_idx2name,
                channel_mean=channel_mean,
                channel_std=channel_std,
                crop_size=crop_size
            )
            
            # Сохраняем и логируем
            saved_path = self._save_figure(fig, "samples", "data_samples")
            
            if saved_path:
                print(f"Примеры изображений залогированы в MLFlow")
            
            plt.close(fig)
            
        except ImportError:
            print(f"Не удалось импортировать show_samples из train.utils")
        except Exception as e:
            print(f"Ошибка при логировании примеров в MLFlow: {e}")
    
    def __del__(self):
        """Деструктор для очистки при удалении объекта"""
        if hasattr(self, '_temp_files') and self._temp_files:
            print(f"MLFlowCallback: очистка {len(self._temp_files)} временных файлов в деструкторе")
            self._cleanup_temp_files()