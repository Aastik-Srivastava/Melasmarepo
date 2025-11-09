"""
Hybrid pipeline utilities:
- Discover multiple segmentation and classification models in media/models
- Load corresponding metrics from *_metrics.pkl
- Select best model by (accuracy + precision) score
- Provide inference helpers (torch + PIL)
"""

__all__ = [
    'discover_models', 'choose_best', 'load_torch_segmentation', 'run_segmentation',
    'overlay_mask', 'load_classifier', 'classify_image_multi', 'init_hybrid_models',
    'SEG_MODEL_OBJ', 'CLS_OBJ', 'DEVICE', 'BEST_SEG', 'BEST_CLS'
]

# Global model instances
SEG_MODEL_OBJ = None
CLS_OBJ = None
DEVICE = 'cpu'
BEST_SEG = None
BEST_CLS = None

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from django.conf import settings


def init_hybrid_models():
    """Initialize global model instances and device setting."""
    global SEG_MODEL_OBJ, CLS_OBJ, DEVICE, BEST_SEG, BEST_CLS

    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Discover and choose best models
    seg_models, cls_models = discover_models()
    BEST_SEG = choose_best(seg_models)
    BEST_CLS = choose_best(cls_models)

    # Load segmentation model
    if BEST_SEG:
        SEG_MODEL_OBJ = load_torch_segmentation(BEST_SEG['path'], DEVICE)

    # Load classification model
    if BEST_CLS:
        CLS_OBJ = load_classifier(BEST_CLS['path'])


def _score_from_metrics(metrics: Dict) -> float:
    acc = float(metrics.get('accuracy', 0.0))
    prec = float(metrics.get('precision', 0.0))
    return acc + prec


def _load_metrics(metrics_path: str) -> Dict:
    try:
        with open(metrics_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def discover_models() -> Tuple[List[Dict], List[Dict]]:
    """Return (segmentation_models, classification_models) discovered under media/models.

    Each entry: {
      'name': str,
      'path': str,
      'metrics': dict
    }
    """
    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    seg_models: List[Dict] = []
    cls_models: List[Dict] = []

    # Heuristics: segmentation -> .pt (torch), classification -> .pkl (sklearn)
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if not os.path.isfile(fpath):
            continue
        base, ext = os.path.splitext(fname)
        if ext.lower() == '.pt':
            metrics_path = os.path.join(models_dir, f"{base}_metrics.pkl")
            metrics = _load_metrics(metrics_path) if os.path.exists(metrics_path) else {}
            seg_models.append({'name': base, 'path': fpath, 'metrics': metrics})
        elif ext.lower() == '.pkl' and not fname.endswith('_metrics.pkl'):
            metrics_path = os.path.join(models_dir, f"{base}_metrics.pkl")
            metrics = _load_metrics(metrics_path) if os.path.exists(metrics_path) else {}
            cls_models.append({'name': base, 'path': fpath, 'metrics': metrics})

    return seg_models, cls_models


def choose_best(models: List[Dict]) -> Optional[Dict]:
    if not models:
        return None
    scored = [(m, _score_from_metrics(m.get('metrics', {}))) for m in models]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def load_torch_segmentation(model_path: str, device: str):
    """Load a torch segmentation model.
    Expects model to be a state_dict for a module that returns logits [B,1,H,W].
    We load it into a simple nn.Module shell if needed by user import elsewhere.
    Here we try a standard torch.load (eager) first; if it fails, attempt state_dict mapping.
    """
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception:
        # Fallback: load into a simple conv head if state_dict compatible with known key
        from .ml_models.segmentation_model import DeepLabV3Plus_TransUNet
        model = DeepLabV3Plus_TransUNet(num_classes=1)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model


def run_segmentation(model, image_pil: Image.Image, device: str = 'cpu', img_size: int = 256) -> np.ndarray:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()[0, 0].cpu().numpy()
    mask_u8 = (mask * 255).astype(np.uint8)
    return mask_u8


def overlay_mask(image_pil: Image.Image, mask_u8: np.ndarray, color=(0, 255, 0), alpha=0.35) -> Image.Image:
    base = image_pil.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray(mask_u8).resize(base.size, Image.NEAREST)
    # Create color layer where mask==255
    color_layer = Image.new('RGBA', base.size, (*color, int(255 * alpha)))
    overlay = Image.composite(color_layer, overlay, mask_img)
    out = Image.alpha_composite(base, overlay).convert('RGB')
    return out


def load_classifier(clf_path: str):
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)
    return clf


def extract_vgg16_features(images_np: np.ndarray) -> np.ndarray:
    """images_np: [N,H,W,3] in RGB, [0,255] or [0,1]. Returns (N,512)."""
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    if images_np.max() <= 1.0:
        images_np = (images_np * 255.0).astype(np.float32)
    arr = preprocess_input(images_np)
    vgg = VGG16(weights='imagenet', include_top=False, pooling='avg')
    feats = vgg.predict(arr, verbose=0)
    return feats.astype(np.float32)


def classify_image_multi(clf, image_pil: Image.Image, roi_mask: Optional[np.ndarray] = None) -> Dict:
    """Classify using sklearn-like model with predict_proba.
    If roi_mask provided, crop to bounding box; otherwise use resized image.
    """
    img = image_pil
    if roi_mask is not None and roi_mask.any():
        # bbox crop
        ys, xs = np.where(roi_mask > 0)
        if len(xs) and len(ys):
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = 10
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(img.width - 1, x1 + pad); y1 = min(img.height - 1, y1 + pad)
            img = img.crop((x0, y0, x1, y1))
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32)[None, ...]
    feats = extract_vgg16_features(arr)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(feats)[0]
    else:
        # fallback to decision function-like
        pred = clf.predict(feats)[0]
        probs = np.array([1.0 - float(pred), float(pred)])
    pred_idx = int(np.argmax(probs))
    return {
        'probs': probs.tolist(),
        'pred_idx': pred_idx
    }


