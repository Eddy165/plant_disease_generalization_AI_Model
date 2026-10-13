from __future__ import annotations
import torch
import torch.nn.functional as F

class GradCAM:
    """Minimal Grad-CAM for ResNet50/MobileNetV2 feature layers."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]
    def _forward_hook(self, module, inputs, output):
        self.activations = output
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), class_idx
    def close(self):
        for handle in self.handles:
            handle.remove()

def get_target_layer(model, model_name):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "custom_cnn":
        return model.features[-2]
    raise ValueError(model_name)
