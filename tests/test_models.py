import torch
from src.models import build_model

def test_custom_cnn_output_shape():
    model = build_model("custom_cnn", 5, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)

def test_mobilenet_output_shape():
    model = build_model("mobilenet_v2", 5, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)
