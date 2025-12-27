import torch.nn as nn
from torchvision import models


class PretrainViT(nn.Module):
    def __init__(self, cfg):
        super(PretrainViT, self).__init__()
        model = models.vit_l_16(weights=cfg.model.model.weights)
        num_classifier_feature = model.heads.head.in_features
        
        model.heads.head = nn.Sequential(
            nn.Linear(num_classifier_feature, cfg.model.model.num_classes)
        )
        self.model = model
        
        if cfg.model.freezing.enabled:
            for name, param in self.model.named_parameters():
                if "heads" not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)