# ML Models Setup Guide

## Overview

MelaScan uses a two-stage ML pipeline:
1. **Segmentation**: Hybrid DeepLabV3+ × TransUNet (Best: Dice 88.63%, IoU 79.62%)
2. **Classification**: Fuzzy C-Means with VGG16 features (Best: ROC-AUC 0.976, Accuracy 92%)

## Model Files Required

Place your trained model files in `media/models/`:

1. **Segmentation Model**: `hybrid_best.pt`
   - Format: PyTorch state dict
   - Architecture: Hybrid DeepLabV3+ × TransUNet
   - Expected performance: Dice ~88%, IoU ~80%

2. **Classification Model**: `fcm_classifier.pkl`
   - Format: Pickled sklearn-compatible classifier
   - Architecture: Fuzzy C-Means with VGG16 features
   - Expected performance: Accuracy ~92%, ROC-AUC ~0.976

## Workflow

1. User uploads an image
2. **Segmentation**: Model finds melasma regions in the image
3. **ROI Extraction**: Extract region of interest from segmented mask
4. **Classification**: Classify ROI as Melasma/Normal using FCM classifier
5. Generate PDF report with results

## Setup Instructions

### Option 1: Using Trained Models (Recommended)

1. **Train models** using the provided notebooks:
   - `hybrid.ipynb` → segmentation model
   - `fuzzy.ipynb` → classification model

2. **Save trained models**:
   ```python
   # For segmentation (from hybrid.ipynb)
   torch.save(model.state_dict(), 'hybrid_best.pt')
   
   # For classification (from fuzzy.ipynb)
   import pickle
   with open('fcm_classifier.pkl', 'wb') as f:
       pickle.dump(best_fcm, f)
   ```

3. **Copy models to media/models/**:
   ```bash
   mkdir -p media/models
   cp hybrid_best.pt media/models/
   cp fcm_classifier.pkl media/models/
   ```

### Option 2: Using Untrained Models (For Testing)

If model files are not available, the system will:
- Use untrained segmentation model (will give random results)
- Use default FCM classifier (needs training)
- Fall back to simple heuristic predictions

## Testing Models

1. Start Django server:
   ```bash
   python manage.py runserver
   ```

2. Upload a test image at `/detect/`

3. Check console for model loading messages:
   - ✅ "Segmentation model loaded on cpu"
   - ✅ "Classification model and VGG16 loaded"
   - ⚠️ "Warning: Could not load..." → models not found

## Model Performance

### Segmentation (Hybrid DeepLabV3+ × TransUNet)
- **Train Dice**: 84.08%
- **Train IoU**: 73.54%
- **Val Dice**: 88.63%
- **Val IoU**: 79.62%
- **Val Specificity**: 92.74%
- **Val Sensitivity**: 89.76%

### Classification (Fuzzy C-Means)
- **Accuracy**: 92%
- **ROC-AUC**: 0.976
- **Precision**: 86% (Others), 86% (Melasma)
- **Recall**: 86% (Others), 100% (Melasma)

## Troubleshooting

### "Segmentation model not found"
- Ensure `hybrid_best.pt` is in `media/models/`
- Check file permissions
- Verify model architecture matches code

### "VGG16 not available"
- Install TensorFlow: `pip install tensorflow==2.15.0`
- VGG16 will download automatically on first use

### "Classification model not found"
- Ensure `fcm_classifier.pkl` is in `media/models/`
- Check that classifier was trained with VGG16 features (512-D)

### Low Prediction Accuracy
- Models may be untrained → train using notebooks
- Model architecture mismatch → retrain with correct config
- Input preprocessing differs → check preprocessing in `ml_integration.py`

## Model Architecture Details

### Segmentation Model
- **Encoder**: ResNet-50 backbone
- **ASPP**: Atrous Spatial Pyramid Pooling
- **Transformer**: Multi-head attention decoder
- **Output**: Binary mask (melasma regions)

### Classification Model
- **Feature Extractor**: VGG16 (ImageNet pretrained)
- **Classifier**: Fuzzy C-Means
- **Input**: 224×224×3 RGB image
- **Output**: Binary classification (Melasma/Normal)

## Performance Optimization

1. **GPU Support**: Install CUDA-enabled PyTorch for faster inference
2. **Model Caching**: Models are cached in memory after first load
3. **Batch Processing**: VGG16 features extracted in batches
4. **Async Processing**: Consider Celery for background processing

