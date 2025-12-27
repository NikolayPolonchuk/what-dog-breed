import torch
import torch.nn.functional as F
from PIL import Image
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import fire


class DogBreedClassifier:
    
    def __init__(self, model_path: str, config_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = self._create_transform()
        
        self.class_names = self._load_class_names()

    def _load_model(self, model_path: str):
        from dog_breed_detector.model.vit_model import PretrainViT
        from omegaconf import DictConfig, OmegaConf
        
        cfg = OmegaConf.create(self.config)
        
        model = PretrainViT(cfg)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model

    def _create_transform(self):
        from torchvision import transforms
        
        cfg = self.config.dataset.preprocessing
        
        return transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.channel_mean,
                std=cfg.channel_std
            )
        ])

    def _load_class_names(self) -> Optional[List[str]]:
        class_names_path = Path('data/class_names.json')
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                return json.load(f)
        return None

    def predict(self, image_path: str, top_k: int = 5) -> Dict:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
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
        with open('data/class_names.json', 'w') as f:
            json.dump(class_names, f, indent=2)
        self.class_names = class_names


class DogBreedCLI:
    def __init__(self, model: str = 'model/model.pth', 
                config: str = 'configs/config.yaml',
                device: str = None):
        self.classifier = DogBreedClassifier(
            model_path=model,
            config_path=config,
            device=device
        )

    def classify(self, image: str, top_k: int = 3):
        result = self.classifier.predict(image, top_k)
        
        print(f"\nImage: {result['image']}")
        print("=" * 50)
        
        for i, pred in enumerate(result['predictions'], 1):
            breed = pred.get('breed', f'Class {pred["class_id"]}')
            print(f"{i}. {breed}: {pred['confidence']} (prob: {pred['probability']:.4f})")
        
        return result

    def batch_classify(self, images: List[str], top_k: int = 3):
        results = self.classifier.predict_batch(images, top_k)
        
        for result in results:
            print(f"\nImage: {result['image']}")
            print("-" * 30)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                for i, pred in enumerate(result['predictions'], 1):
                    breed = pred.get('breed', f'Class {pred["class_id"]}')
                    print(f"{i}. {breed}: {pred['confidence']}")
        
        return results

    def info(self):
        total_params = sum(p.numel() for p in self.classifier.model.parameters())
        trainable_params = sum(p.numel() for p in self.classifier.model.parameters() 
                            if p.requires_grad)
        
        print(f"Device: {self.classifier.device}")
        print(f"Model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Class names loaded: {self.classifier.class_names is not None}")
        if self.classifier.class_names:
            print(f"Number of classes: {len(self.classifier.class_names)}")


def main():
    fire.Fire(DogBreedCLI)

if __name__ == "__main__":
    main()