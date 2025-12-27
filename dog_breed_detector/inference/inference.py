import torch
import torch.nn.functional as F
from PIL import Image
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional


class DogBreedClassifier:
    """Классификатор пород собак для продакшн инференса"""
    
    def __init__(self, model_path: str, config_path: str, device: str = None):
        """
        Инициализация классификатора
        
        Args:
            model_path: путь к файлу с весами модели (.pth)
            config_path: путь к конфигурационному файлу
            device: устройство для вычислений (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Загрузка модели
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Настройка трансформаций
        self.transform = self._create_transform()
        
        # Загрузка названий классов (если есть)
        self.class_names = self._load_class_names()
    
    def _load_model(self, model_path: str):
        """Загрузка архитектуры модели и весов"""
        # Импорт здесь чтобы избежать зависимостей при использовании
        from dog_breed_detector.model.vit_model import PretrainViT
        from omegaconf import DictConfig, OmegaConf
        
        # Конвертация конфига в формат OmegaConf
        cfg = OmegaConf.create(self.config)
        
        # Создание модели
        model = PretrainViT(cfg)
        
        # Загрузка весов
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _create_transform(self):
        """Создание трансформаций для инференса"""
        from torchvision import transforms
        
        cfg = self.config['dataset']['preprocessing']
        
        return transforms.Compose([
            transforms.Resize(cfg['resize']),
            transforms.CenterCrop(cfg['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg['channel_mean'],
                std=cfg['channel_std']
            )
        ])
    
    def _load_class_names(self) -> Optional[List[str]]:
        """Загрузка названий классов"""
        # Попробуем найти файл с названиями классов
        class_names_path = Path('data/class_names.json')
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                return json.load(f)
        return None
    
    def predict(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Предсказание породы собаки
        
        Args:
            image_path: путь к изображению
            top_k: количество топовых предсказаний
            
        Returns:
            Словарь с предсказаниями
        """
        # Загрузка и предобработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Получение топ-K предсказаний
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Форматирование результата
        result = {
            'image': image_path,
            'predictions': []
        }
        
        for i in range(top_k):
            class_idx = top_indices[0, i].item()
            probability = top_probs[0, i].item()
            
            prediction = {
                'class_id': class_idx,
                'probability': float(probability),
                'confidence': f"{probability:.1%}"
            }
            
            if self.class_names and class_idx < len(self.class_names):
                prediction['breed'] = self.class_names[class_idx]
            
            result['predictions'].append(prediction)
        
        return result
    
    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[Dict]:
        """Пакетное предсказание"""
        results = []
        for img_path in image_paths:
            try:
                results.append(self.predict(img_path, top_k))
            except Exception as e:
                results.append({
                    'image': img_path,
                    'error': str(e),
                    'predictions': []
                })
        return results
    
    def save_class_names(self, class_names: List[str]):
        """Сохранение названий классов"""
        with open('data/class_names.json', 'w') as f:
            json.dump(class_names, f, indent=2)
        self.class_names = class_names


def main():
    """Пример использования классификатора"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dog Breed Classifier Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='model/model.pth', help='Model path')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config path')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Инициализация классификатора
    classifier = DogBreedClassifier(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Предсказание
    result = classifier.predict(args.image, args.top_k)
    
    # Вывод результата
    print(f"\nImage: {result['image']}")
    print("=" * 50)
    
    for i, pred in enumerate(result['predictions'], 1):
        breed = pred.get('breed', f'Class {pred["class_id"]}')
        print(f"{i}. {breed}: {pred['confidence']} (prob: {pred['probability']:.4f})")


if __name__ == "__main__":
    main()