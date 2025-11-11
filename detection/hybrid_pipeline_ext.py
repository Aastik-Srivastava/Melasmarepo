import os
import numpy as np
from PIL import Image
from django.conf import settings

from . import hybrid_pipeline as hp


def process_segmentation(image_path, model_name=None):
    """Process image for segmentation only.

    Opens the saved image (absolute path), ensures the segmentation model is
    initialized (lazy), runs segmentation, saves an overlay under
    MEDIA_ROOT/reports and returns metadata.
    """
    try:
        # Optionally load a specific segmentation model by name. If none
        # provided, use the global SEG_MODEL_OBJ (initialized lazily).
        if model_name:
            seg_models, _ = hp.discover_models()
            selected = next((m for m in seg_models if m.get('name') == model_name), None)
            if not selected:
                return {'error': f'Segmentation model "{model_name}" not found'}
            # load the selected model (map to current device)
            device = hp.DEVICE or ('cuda' if hasattr(hp, 'torch') and hp.torch.cuda.is_available() else 'cpu')
            model = hp.load_torch_segmentation(selected['path'], device)
        else:
            # Ensure models are initialized lazily
            if not hp.SEG_MODEL_OBJ:
                hp.init_hybrid_models()
            if not hp.SEG_MODEL_OBJ:
                return {'error': 'No segmentation model available'}
            model = hp.SEG_MODEL_OBJ

        image = Image.open(image_path).convert('RGB')
        mask_u8 = hp.run_segmentation(model, image, device=hp.DEVICE, img_size=256)
        overlay = hp.overlay_mask(image, mask_u8, color=(0, 255, 0), alpha=0.35)

        # Save overlay and predicted mask
        overlays_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
        os.makedirs(overlays_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save overlay
        overlay_name = base_name + '_overlay.jpg'
        overlay_path = os.path.join(overlays_dir, overlay_name)
        overlay.save(overlay_path, format='JPEG', quality=92)
        
        # Save predicted mask (grayscale)
        pred_mask_name = base_name + '_predicted_mask.png'
        pred_mask_path = os.path.join(overlays_dir, pred_mask_name)
        Image.fromarray(mask_u8).save(pred_mask_path)
        
        # Check for ground truth mask (only exists for validation/test images)
        # Try common naming patterns from notebooks: {name}_mask.png, {name}_gt.png
        possible_gt_names = [
            base_name + '_mask.png',
            base_name + '_gt.png',
            base_name.replace('_result', '') + '_mask.png',  # handle result suffix
        ]
        gt_mask_path = None
        # Look in masks directory or same directory as image
        search_dirs = [
            os.path.join(settings.MEDIA_ROOT, 'masks'),
            os.path.dirname(image_path),
            overlays_dir,
        ]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for gt_name in possible_gt_names:
                potential_gt = os.path.join(search_dir, gt_name)
                if os.path.exists(potential_gt):
                    gt_mask_path = potential_gt
                    break
            if gt_mask_path:
                break

        # Calculate affected area percentage
        total_pixels = mask_u8.size
        affected_pixels = np.sum(mask_u8 > 128)  # assuming binary mask
        affected_percentage = (affected_pixels / total_pixels) * 100 if total_pixels else 0.0

        return {
            'result': 'Segmentation Complete',
            'overlay_path': overlay_path,
            'ground_truth_mask_path': gt_mask_path,  # May be None for new uploads
            'predicted_mask_path': pred_mask_path,
            'model_used': (model_name or (hp.BEST_SEG.get('name') if hp.BEST_SEG else 'Unknown')),
            'affected_percentage': affected_percentage,
            'confidence': None  # No confidence score for segmentation
        }
    except Exception as e:
        return {'error': str(e)}


def process_classification(image_path, model_name=None):
    """Process image for classification only.

    Ensures classifier is loaded (lazy), optionally computes ROI from a
    segmentation model if available, then runs classification and returns
    prediction metadata.
    """
    try:
        # Optionally load a specific classifier by name. If none provided,
        # use the global CLS_OBJ (initialized lazily).
        if model_name:
            _, cls_models = hp.discover_models()
            selected = next((m for m in cls_models if m.get('name') == model_name), None)
            if not selected:
                return {'error': f'Classifier "{model_name}" not found'}
            clf = hp.load_classifier(selected['path'])
        else:
            if not hp.CLS_OBJ:
                hp.init_hybrid_models()
            if not hp.CLS_OBJ:
                return {'error': 'No classification model available'}
            clf = hp.CLS_OBJ

        image = Image.open(image_path).convert('RGB')
        roi_mask = None

        # Optionally use segmentation ROI if segmentation model is loaded
        if hp.SEG_MODEL_OBJ:
            mask_u8 = hp.run_segmentation(hp.SEG_MODEL_OBJ, image, device=hp.DEVICE, img_size=256)
            roi_mask = np.array(Image.fromarray(mask_u8).resize(image.size, Image.NEAREST))

        result = hp.classify_image_multi(clf, image, roi_mask=roi_mask)
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
            'model_used': (model_name or (hp.BEST_CLS.get('name') if hp.BEST_CLS else 'Unknown')),
            'confidence': confidence * 100,  # percent
            'probabilities': probs
        }
    except Exception as e:
        return {'error': str(e)}