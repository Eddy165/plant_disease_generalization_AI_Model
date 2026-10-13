from __future__ import annotations
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("custom_cnn", "resnet50", "mobilenet_v2")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(in_c, out_c, pool=True):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers
        self.features = nn.Sequential(
            *block(3, 32), *block(32, 64), *block(64, 128),
            *block(128, 256), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(name, num_classes, pretrained=True):
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}")
