import os
import numpy as np
from PIL import Image
from django.conf import settings

from . import hybrid_pipeline as hp


def process_segmentation(image_path):
    """Process image for segmentation only.

    Opens the saved image (absolute path), ensures the segmentation model is
    initialized (lazy), runs segmentation, saves an overlay under
    MEDIA_ROOT/reports and returns metadata.
    """
    try:
        # Ensure models are initialized lazily
        if not hp.SEG_MODEL_OBJ:
            hp.init_hybrid_models()
        if not hp.SEG_MODEL_OBJ:
            return {'error': 'No segmentation model available'}

        image = Image.open(image_path).convert('RGB')
        mask_u8 = hp.run_segmentation(hp.SEG_MODEL_OBJ, image, device=hp.DEVICE, img_size=256)
        overlay = hp.overlay_mask(image, mask_u8, color=(0, 255, 0), alpha=0.35)

        # Save overlay
        overlays_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
        os.makedirs(overlays_dir, exist_ok=True)
        overlay_name = os.path.splitext(os.path.basename(image_path))[0] + '_overlay.jpg'
        overlay_path = os.path.join(overlays_dir, overlay_name)
        overlay.save(overlay_path, format='JPEG', quality=92)

        # Calculate affected area percentage
        total_pixels = mask_u8.size
        affected_pixels = np.sum(mask_u8 > 128)  # assuming binary mask
        affected_percentage = (affected_pixels / total_pixels) * 100 if total_pixels else 0.0

        return {
            'result': 'Segmentation Complete',
            'overlay_path': overlay_path,
            'model_used': hp.BEST_SEG.get('name', 'Unknown') if hp.BEST_SEG else 'Unknown',
            'affected_percentage': affected_percentage,
            'confidence': None  # No confidence score for segmentation
        }
    except Exception as e:
        return {'error': str(e)}


def process_classification(image_path):
    """Process image for classification only.

    Ensures classifier is loaded (lazy), optionally computes ROI from a
    segmentation model if available, then runs classification and returns
    prediction metadata.
    """
    try:
        if not hp.CLS_OBJ:
            hp.init_hybrid_models()
        if not hp.CLS_OBJ:
            return {'error': 'No classification model available'}

        image = Image.open(image_path).convert('RGB')
        roi_mask = None

        # Optionally use segmentation ROI if segmentation model is loaded
        if hp.SEG_MODEL_OBJ:
            mask_u8 = hp.run_segmentation(hp.SEG_MODEL_OBJ, image, device=hp.DEVICE, img_size=256)
            roi_mask = np.array(Image.fromarray(mask_u8).resize(image.size, Image.NEAREST))

        result = hp.classify_image_multi(hp.CLS_OBJ, image, roi_mask=roi_mask)
        pred_idx = result['pred_idx']
        probs = result['probs']

        # Get prediction and confidence
        if len(probs) > 2:  # Multi-class case
            prediction = f"Type {pred_idx}" if pred_idx != (len(probs) - 1) else "Melasma"
            confidence = float(probs[pred_idx])
        else:  # Binary case
            prediction = "Melasma" if pred_idx == 1 else "No Melasma"
            confidence = float(probs[pred_idx])

        return {
            'result': prediction,
            'model_used': hp.BEST_CLS.get('name', 'Unknown') if hp.BEST_CLS else 'Unknown',
            'confidence': confidence * 100,  # percent
            'probabilities': probs
        }
    except Exception as e:
        return {'error': str(e)}