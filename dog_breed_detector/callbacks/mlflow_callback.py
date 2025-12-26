import mlflow
import mlflow.pytorch
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
import tempfile
import os


class MLflowCallback(Callback):
    """MLflow callback для PyTorch Lightning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlflow_config = config.get('mlflow', {})
        self.enabled = self.mlflow_config.get('enabled', False)
        self.experiment_id = None
        self.run_id = None
        
        if self.enabled:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Настройка MLflow"""
        tracking_uri = self.mlflow_config.get('tracking_uri', './mlruns')
        experiment_name = self.mlflow_config.get('experiment_name', 'default')
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Создание или получение эксперимента
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        print(f"MLflow настроен:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def on_fit_start(self, trainer, pl_module):
        """Начало обучения"""
        if not self.enabled:
            return
        
        # Старт run
        run_name = self.mlflow_config.get('run_name', 'pytorch-lightning-run')
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        self.run_id = mlflow.active_run().info.run_id
        
        # Логирование параметров
        if self.mlflow_config.get('logging', {}).get('params', True):
            self._log_params(pl_module)
        
        print(f"MLflow run начат: {run_name} (ID: {self.run_id})")
    
    def _log_params(self, pl_module):
        """Логирование параметров"""
        try:
            # Логирование конфигурации
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            
            # Сохраняем конфиг как YAML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                config_path = f.name
            
            mlflow.log_artifact(config_path, "config")
            os.unlink(config_path)
            
            # Логирование гиперпараметров модели
            if hasattr(pl_module, 'hparams'):
                params = {}
                for key, value in pl_module.hparams.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        params[key] = value
                    elif isinstance(value, (list, tuple, dict)):
                        params[key] = str(value)
                
                mlflow.log_params(params)
            
            print(f"Параметры залогированы в MLflow")
            
        except Exception as e:
            print(f"Ошибка при логировании параметров в MLflow: {e}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Конец эпохи обучения"""
        if not self.enabled:
            return
        
        # Логирование метрик
        if self.mlflow_config.get('logging', {}).get('metrics', True):
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
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Конец эпохи валидации"""
        if not self.enabled:
            return
        
        # Логирование графиков
        if (self.mlflow_config.get('logging', {}).get('plots', True) and 
            hasattr(pl_module, 'history')):
            
            # Создаем график истории обучения
            history = pl_module.history
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # График loss
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
            axes[0].plot(epochs, history['valid_loss'], 'r-', label='Valid Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # График accuracy
            if 'train_accuracy' in history and 'valid_accuracy' in history:
                axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy')
                axes[1].plot(epochs, history['valid_accuracy'], 'r-', label='Valid Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Training and Validation Accuracy')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем и логируем график
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, "plots")
                os.unlink(f.name)
            
            plt.close()
    
    def on_train_end(self, trainer, pl_module):
        """Конец обучения"""
        if not self.enabled:
            return
        
        # Логирование модели
        if self.mlflow_config.get('logging', {}).get('models', True):
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=pl_module,
                    artifact_path="model",
                    registered_model_name=f"dog-breed-vit-{self.run_id}"
                )
                print(f"Модель залогирована в MLflow")
            except Exception as e:
                print(f"Ошибка при логировании модели в MLflow: {e}")
        
        # Логирование чекпоинтов
        if (self.mlflow_config.get('artifacts', {}).get('save_checkpoints', True) and
            hasattr(trainer, 'checkpoint_callback') and
            trainer.checkpoint_callback is not None):
            
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path:
                mlflow.log_artifact(best_model_path, "checkpoints")
        
        # Завершение run
        mlflow.end_run()
        print(f"MLflow run завершен (ID: {self.run_id})")
    
    def log_samples(self, batch_img, batch_label, label_idx2name, channel_mean, channel_std, crop_size):
        """Логирование примеров изображений"""
        if not self.enabled:
            return
        
        if not self.mlflow_config.get('artifacts', {}).get('save_samples', True):
            return
        
        try:
            from train.utils import show_samples
            
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
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, "samples")
                os.unlink(f.name)
            
            plt.close(fig)
            print(f"Примеры изображений залогированы в MLflow")
            
        except Exception as e:
            print(f"Ошибка при логировании примеров в MLflow: {e}")