# demo/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from dataset import VAL_TRANSFORMS
from forensic_text import generate_forensic_report
from gradcam import GradCAM, get_top_zones
from model import DeepfakeClassifier

try:
    from facenet_pytorch import MTCNN as _MTCNN
    _mtcnn = _MTCNN(image_size=224, margin=20, keep_all=False, device='cpu')
except Exception:
    _mtcnn = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = Path(__file__).parent.parent / 'checkpoints' / 'best_model.pt'

model = DeepfakeClassifier(freeze_blocks=5)
if CHECKPOINT.exists():
    model.load_state_dict(torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True))
model = model.to(DEVICE).eval()
grad_cam = GradCAM(model)


def _crop_face(pil_img: Image.Image):
    """Returns (224x224 face crop as PIL, crop tensor) or (None, None) if no face."""
    if _mtcnn is None:
        resized = pil_img.resize((224, 224))
        tensor = VAL_TRANSFORMS(resized).unsqueeze(0).to(DEVICE)
        return resized, tensor

    crop_tensor = _mtcnn(pil_img)
    if crop_tensor is None:
        return None, None

    crop_np = ((crop_tensor.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    crop_pil = Image.fromarray(crop_np)
    img_tensor = VAL_TRANSFORMS(crop_pil).unsqueeze(0).to(DEVICE)
    return crop_pil, img_tensor


def analyze_image(pil_img: Image.Image):
    if pil_img is None:
        return None, "No image provided.", "Upload an image to begin."

    crop_pil, img_tensor = _crop_face(pil_img)
    if crop_pil is None:
        return None, "No face detected.", "Unable to analyze — no face found in the image."

    heatmap, confidence = grad_cam.compute(img_tensor)
    overlay = grad_cam.overlay(crop_pil, heatmap)
    zones = get_top_zones(heatmap, top_k=2)
    report = generate_forensic_report(confidence, zones[0], zones[1])
    label = "FAKE" if confidence >= 0.5 else "REAL"
    verdict = f"{label}  —  Confidence: {confidence:.1%}"
    return overlay, verdict, report


def analyze_video(video_path: str):
    if video_path is None:
        return None, "No video provided.", "Upload a video to begin."

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None, "Could not read video.", "File may be corrupt or unsupported format."

    indices = [int(i * total / 15) for i in range(15)]
    all_confidences = []
    best_conf, best_overlay, best_report = 0.0, None, ""

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        crop_pil, img_tensor = _crop_face(pil_frame)
        if crop_pil is None:
            continue
        heatmap, conf = grad_cam.compute(img_tensor)
        overlay = grad_cam.overlay(crop_pil, heatmap)
        zones = get_top_zones(heatmap, top_k=2)
        report = generate_forensic_report(conf, zones[0], zones[1])
        all_confidences.append(conf)
        if conf > best_conf:
            best_conf, best_overlay, best_report = conf, overlay, report

    cap.release()

    if not all_confidences:
        return None, "No faces detected in video.", "Could not analyze any frames."

    mean_conf = np.mean(all_confidences)
    label = "FAKE" if mean_conf >= 0.5 else "REAL"
    verdict = f"[VIDEO] {label}  —  Mean confidence: {mean_conf:.1%} over {len(all_confidences)} frames"
    return best_overlay, verdict, best_report


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("# Deepfake Detection with Forensic Explainability")
    gr.Markdown(
        "Upload an image or short video. The model detects manipulation artifacts, "
        "highlights suspicious facial regions with Grad-CAM, and generates a forensic report."
    )

    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Input Image")
            img_overlay = gr.Image(label="Grad-CAM Heatmap")
        img_verdict = gr.Textbox(label="Verdict", interactive=False)
        img_report  = gr.Textbox(label="Forensic Report", lines=4, interactive=False)
        gr.Button("Analyze Image").click(
            analyze_image,
            inputs=img_in,
            outputs=[img_overlay, img_verdict, img_report],
        )

    with gr.Tab("Video"):
        vid_in = gr.Video(label="Input Video (≤ 60 s recommended)")
        with gr.Row():
            vid_overlay = gr.Image(label="Highest-Confidence Frame — Grad-CAM")
            with gr.Column():
                vid_verdict = gr.Textbox(label="Video Verdict", interactive=False)
                vid_report  = gr.Textbox(label="Forensic Report", lines=4, interactive=False)
        gr.Button("Analyze Video").click(
            analyze_video,
            inputs=vid_in,
            outputs=[vid_overlay, vid_verdict, vid_report],
        )

if __name__ == "__main__":
    demo.launch()
