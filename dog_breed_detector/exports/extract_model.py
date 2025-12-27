import torch
import torch.serialization
from omegaconf import DictConfig
torch.serialization.add_safe_globals([DictConfig])
from dog_breed_detector.model.vit_model import PretrainViT
from dog_breed_detector.train.train import LitDogModel
import hydra


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def convert_model(cfg: DictConfig):
    """Загружает Lightning модель и сохраняет чистые веса"""
    
    ckpt_path = "checkpoints/last.ckpt"  # или путь к вашему .ckpt
    
    # Загружаем Lightning модель
    print(f"Загружаем {ckpt_path}...")
    model = LitDogModel.load_from_checkpoint(ckpt_path, cfg=cfg, weights_only=False)
    
    # Получаем state_dict
    state_dict = model.model.state_dict()  # Доступ к внутренней модели
    
    # Сохраняем как .pth
    output_path = "model/model.pth"
    torch.save(state_dict, output_path)
    
    print(f" Модель сохранена в: {output_path}")
    print(f" Размер модели: {sum(p.numel() for p in state_dict.values()):,} параметров")
    
    # 4. Тест загрузки
    print("\nТест загрузки:")
    test_model = PretrainViT(cfg)
    test_model.load_state_dict(torch.load(output_path))
    print("Тест загрузки пройден успешно!")

if __name__ == "__main__":
    convert_model()