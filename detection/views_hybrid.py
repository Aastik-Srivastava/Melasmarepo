from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings

import os
from PIL import Image
import numpy as np

from .hybrid_pipeline import (
    discover_models,
    choose_best,
    load_torch_segmentation,
    run_segmentation,
    overlay_mask,
    load_classifier,
    classify_image_multi,
)


# Global caches
SEG_MODELS = []
CLS_MODELS = []
BEST_SEG = None
BEST_CLS = None
SEG_MODEL_OBJ = None
CLS_OBJ = None
DEVICE = 'cpu'


def init_hybrid_models():
    global SEG_MODELS, CLS_MODELS, BEST_SEG, BEST_CLS, SEG_MODEL_OBJ, CLS_OBJ, DEVICE
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEG_MODELS, CLS_MODELS = discover_models()
    BEST_SEG = choose_best(SEG_MODELS)
    BEST_CLS = choose_best(CLS_MODELS)
    if BEST_SEG:
        print(f"[Hybrid] Best segmentation model: {BEST_SEG['name']} (score={BEST_SEG.get('metrics',{})})")
        SEG_MODEL_OBJ = load_torch_segmentation(BEST_SEG['path'], DEVICE)
    else:
        print("[Hybrid] No segmentation models found.")
    if BEST_CLS:
        print(f"[Hybrid] Best classification model: {BEST_CLS['name']} (score={BEST_CLS.get('metrics',{})})")
        CLS_OBJ = load_classifier(BEST_CLS['path'])
    else:
        print("[Hybrid] No classification models found.")


# Initialize on import
try:
    init_hybrid_models()
except Exception as e:
    print(f"[Hybrid] Initialization error: {e}")


def hybrid_index(request):
    return render(request, 'detection/hybrid_index.html', {
        'best_seg': BEST_SEG,
        'best_cls': BEST_CLS,
    })


@require_http_methods(["POST"])
def hybrid_run_segmentation(request):
    if not SEG_MODEL_OBJ:
        messages.error(request, 'No segmentation model available.')
        return hybrid_index(request)

    file = request.FILES.get('image')
    if not file:
        messages.error(request, 'Please upload an image.')
        return hybrid_index(request)

    fs = FileSystemStorage()
    filename = fs.save(os.path.join('uploads', file.name), file)
    file_url = fs.url(filename)
    abs_path = fs.path(filename)

    image = Image.open(abs_path).convert('RGB')
    mask_u8 = run_segmentation(SEG_MODEL_OBJ, image, device=DEVICE, img_size=256)
    overlay = overlay_mask(image, mask_u8, color=(0, 255, 0), alpha=0.35)

    # Save overlay
    overlays_dir = os.path.join(settings.MEDIA_ROOT, 'overlays')
    os.makedirs(overlays_dir, exist_ok=True)
    overlay_name = os.path.splitext(os.path.basename(filename))[0] + '_overlay.jpg'
    overlay_path = os.path.join(overlays_dir, overlay_name)
    overlay.save(overlay_path, format='JPEG', quality=92)
    overlay_url = os.path.join(settings.MEDIA_URL, 'overlays', overlay_name)

    messages.success(request, f"Segmentation done using: {BEST_SEG['name'] if BEST_SEG else 'N/A'}")
    return render(request, 'detection/hybrid_index.html', {
        'best_seg': BEST_SEG,
        'best_cls': BEST_CLS,
        'uploaded_image_url': file_url,
        'overlay_url': overlay_url,
    })


@require_http_methods(["POST"])
def hybrid_run_classification(request):
    if not CLS_OBJ:
        messages.error(request, 'No classification model available.')
        return hybrid_index(request)

    # Use segmented result if provided; else original image
    file = request.FILES.get('image')
    seg_file = request.FILES.get('segmented')  # optional pre-segmented image

    fs = FileSystemStorage()
    image_pil = None
    roi_mask = None

    if seg_file:
        seg_name = fs.save(os.path.join('uploads', seg_file.name), seg_file)
        seg_path = fs.path(seg_name)
        image_pil = Image.open(seg_path).convert('RGB')
    elif file:
        img_name = fs.save(os.path.join('uploads', file.name), file)
        img_path = fs.path(img_name)
        image_pil = Image.open(img_path).convert('RGB')
        # try to segment quickly with best seg if available
        if SEG_MODEL_OBJ:
            mask_u8 = run_segmentation(SEG_MODEL_OBJ, image_pil, device=DEVICE, img_size=256)
            roi_mask = np.array(Image.fromarray(mask_u8).resize(image_pil.size, Image.NEAREST))
    else:
        messages.error(request, 'Please upload an image.')
        return hybrid_index(request)

    result = classify_image_multi(CLS_OBJ, image_pil, roi_mask=roi_mask)
    pred_idx = result['pred_idx']
    probs = result['probs']

    label = 'melasma' if pred_idx == 1 else 'non-melasma'
    non_melasma_type = None
    # If multi-class (>2), assume index 0..k-1 maps to specific non-melasma subtypes; last index melasma
    if len(probs) > 2 and pred_idx != (len(probs) - 1):
        non_melasma_type = f"non-melasma type #{pred_idx}"
    elif pred_idx == 0:
        non_melasma_type = 'Others'

    messages.success(request, f"Classification using: {BEST_CLS['name'] if BEST_CLS else 'N/A'}")
    return render(request, 'detection/hybrid_index.html', {
        'best_seg': BEST_SEG,
        'best_cls': BEST_CLS,
        'classification_label': label,
        'classification_probs': probs,
        'non_melasma_type': non_melasma_type,
    })


