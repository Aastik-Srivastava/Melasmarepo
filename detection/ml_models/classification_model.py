"""
Fuzzy C-Means Classification Model
Best performance: ROC-AUC 0.976, Accuracy 92%
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import pickle
import os


class FCMClassifier(BaseEstimator, ClassifierMixin):
    """
    Fuzzy C-Means classifier for melasma classification.
    Uses VGG16 features (512-D) from pretrained ImageNet model.
    """

    def __init__(self, m=2.0, k0=2, k1=2, max_iter=150, tol=1e-4, scale=True, random_state=42):
        self.m = float(m)
        self.k0 = int(k0)
        self.k1 = int(k1)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = bool(scale)
        self.random_state = int(random_state)

    def _init_scaler(self, X):
        if self.scale:
            self._scaler = StandardScaler().fit(X)
            return self._scaler.transform(X)
        else:
            self._scaler = None
            return X

    def _transform(self, X):
        if getattr(self, "_scaler", None) is not None:
            return self._scaler.transform(X)
        return X

    @staticmethod
    def _fcm(X, C, m, max_iter, tol, rnd):
        """
        Basic FCM on data X into C clusters.
        Returns centers (C,D). Based on standard objective.
        """
        rng = np.random.RandomState(rnd)
        N, D = X.shape
        # initialize memberships randomly, rows sum to 1
        U = rng.rand(N, C)
        U = U / (U.sum(axis=1, keepdims=True) + 1e-12)

        for _ in range(max_iter):
            # cluster centers
            Um = U ** m
            centers = (Um.T @ X) / (Um.sum(axis=0, keepdims=True).T + 1e-12)

            # distances to centers (N,C)
            d2 = np.maximum(
                np.sum((X[:,None,:] - centers[None,:,:])**2, axis=2),
                1e-12
            )
            # update membership
            inv = (d2 ** (-1.0/(m-1)))
            U_new = inv / (np.sum(inv, axis=1, keepdims=True) + 1e-12)

            # check change
            if np.linalg.norm(U_new - U) < tol:
                U = U_new
                break
            U = U_new

        return centers, U

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0,1], dtype=int)  # assumed binary
        Xs = self._init_scaler(X)

        # split by class
        X0, X1 = Xs[y==0], Xs[y==1]
        if len(X0)==0 or len(X1)==0:
            raise ValueError("Both classes must be present in training data.")

        # run FCM separately per class → get prototypes & exponents
        self.C0_, self.U0_ = self._fcm(X0, self.k0, self.m, self.max_iter, self.tol, self.random_state)
        self.C1_, self.U1_ = self._fcm(X1, self.k1, self.m, self.max_iter, self.tol, self.random_state+1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        Z = self._transform(X)
        # memberships to class-0 prototypes
        d0 = np.maximum(np.sum((Z[:,None,:]-self.C0_[None,:,:])**2, axis=2), 1e-12)
        w0 = (d0 ** (-1.0/(self.m-1)))  # (N,k0)
        s0 = w0.sum(axis=1)             # per-sample class-0 support

        # memberships to class-1 prototypes
        d1 = np.maximum(np.sum((Z[:,None,:]-self.C1_[None,:,:])**2, axis=2), 1e-12)
        w1 = (d1 ** (-1.0/(self.m-1)))  # (N,k1)
        s1 = w1.sum(axis=1)             # per-sample class-1 support

        S = np.vstack([s0, s1]).T + 1e-12
        probs = S / S.sum(axis=1, keepdims=True)
        # order: [class0, class1]
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:,1] >= 0.5).astype(int)


def load_classification_model(model_path=None, vgg_model=None):
    """
    Load Fuzzy C-Means classification model.
    
    Args:
        model_path: Path to saved model (.pkl file)
        vgg_model: VGG16 feature extractor (optional, will create if None)
    
    Returns:
        Tuple of (classifier, vgg_model)
    """
    # Load VGG16 for feature extraction if not provided
    if vgg_model is None:
        try:
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
            vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            print("Loaded VGG16 for feature extraction")
        except ImportError:
            print("Warning: TensorFlow not available. VGG16 features cannot be extracted.")
            return None, None
    
    # Load classifier if path provided
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            print(f"Loaded classification model from {model_path}")
            return classifier, vgg_model
        except Exception as e:
            print(f"Warning: Could not load classifier: {e}. Using default untrained model.")
    
    # Return default untrained model
    classifier = FCMClassifier(m=1.5, k0=1, k1=1, scale=True, random_state=42)
    print("Warning: Using untrained FCM classifier. Model needs to be trained first.")
    return classifier, vgg_model


def extract_vgg_features(image_array, vgg_model, batch_size=32):
    """
    Extract VGG16 features from image array.
    
    Args:
        image_array: Numpy array of images [N, H, W, 3] normalized [0,1]
        vgg_model: VGG16 model
        batch_size: Batch size for processing
    
    Returns:
        Feature array [N, 512]
    """
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    # Preprocess for VGG16 (ImageNet normalization)
    # VGG expects [0,255] range, so scale back up
    if image_array.max() <= 1.0:
        image_array = (image_array * 255.0).astype(np.float32)
    
    # Apply VGG preprocessing
    processed = preprocess_input(image_array)
    
    # Extract features
    features = vgg_model.predict(processed, verbose=0)
    return features.astype(np.float32)


def classify_image(classifier, image_array, vgg_model, threshold=0.5):
    """
    Classify an image as melasma or normal.
    
    Args:
        classifier: FCMClassifier instance
        image_array: Preprocessed image array [1, H, W, 3]
        vgg_model: VGG16 feature extractor
        threshold: Classification threshold
    
    Returns:
        Dictionary with classification result
    """
    # Extract features
    features = extract_vgg_features(image_array, vgg_model)
    
    # Predict
    probs = classifier.predict_proba(features)
    prob_melasma = probs[0, 1]  # Probability of melasma class
    pred = int(prob_melasma >= threshold)
    
    return {
        'prediction': pred,
        'probability_melasma': float(prob_melasma),
        'probability_normal': float(probs[0, 0]),
        'confidence': float(max(prob_melasma, probs[0, 0])),
    }

