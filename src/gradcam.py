# src/gradcam.py
import cv2
import numpy as np
import torch
from PIL import Image

ZONE_NAMES = ["forehead", "eyes", "nose", "jaw", "hairline"]
ZONE_ROWS  = [(0, 60), (60, 100), (100, 145), (145, 185), (185, 224)]


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.features[-1]

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd_hook))
        self._hooks.append(target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, img_tensor: torch.Tensor):
        """
        img_tensor: (1, 3, 224, 224) FloatTensor on same device as model.
        Returns: (heatmap ndarray (224, 224) in [0,1], confidence float)
        """
        self.model.eval()
        output = self.model(img_tensor)           # (1, 1) logits
        confidence = torch.sigmoid(output).item()

        self.model.zero_grad()
        output.backward()

        # Global average pool gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, confidence

    def overlay(self, img_pil: Image.Image, heatmap: np.ndarray) -> Image.Image:
        """Blend jet colormap heatmap over PIL image at 50% opacity."""
        img_np = np.array(img_pil.resize((224, 224)))
        heat_u8 = (heatmap * 255).astype(np.uint8)
        jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
        blended = (0.5 * img_np + 0.5 * jet_rgb).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)


def get_top_zones(heatmap: np.ndarray, top_k: int = 2) -> list:
    """Return names of top_k zones ranked by mean Grad-CAM activation."""
    scores = [
        (name, heatmap[r0:r1, :].mean())
        for name, (r0, r1) in zip(ZONE_NAMES, ZONE_ROWS)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]
