"""StablePNN implementation extracted from MLMODELS/PNNfinall.ipynb

This module provides a StablePNN classifier compatible with the pickled
objects created during research. Having this class importable allows
unpickling without recreating the pickle files.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class StablePNN(BaseEstimator, ClassifierMixin):

    def __init__(self, sigma=1.0, priors='empirical', batch=512):
        self.sigma = float(sigma)
        self.priors = priors
        self.batch = batch

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        # store per-class matrices
        self._by_class = {c: X[y == c] for c in self.classes_}
        # priors
        if self.priors == 'empirical':
            self._log_prior = {c: np.log(len(self._by_class[c]) / len(X) + 1e-12) for c in self.classes_}
        else:
            self._log_prior = {c: -np.log(len(self.classes_)) for c in self.classes_}
        self._inv_sigma2 = 1.0 / (self.sigma ** 2 + 1e-12)
        return self

    def _log_scores(self, X):
        """Return log posterior up to a constant: log P(c) + log sum_i exp(-0.5 d2 / sigma^2)."""
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        C = len(self.classes_)
        logs = np.full((N, C), -np.inf, dtype=np.float64)

        for j, c in enumerate(self.classes_):
            Xc = self._by_class[c]  # (Nc, D)
            if Xc.size == 0:
                continue
            # Accumulate log-sum-exp over class samples in batches to save RAM
            acc_log = np.full(N, -np.inf, dtype=np.float64)
            for i0 in range(0, N, self.batch):
                xb = X[i0:i0 + self.batch]  # (B, D)
                xb2 = np.sum(xb * xb, axis=1, keepdims=True)           # (B,1)
                xc2 = np.sum(Xc * Xc, axis=1, keepdims=True).T         # (1,Nc)
                d2  = xb2 + xc2 - 2.0 * xb.dot(Xc.T)                   # (B,Nc)
                np.maximum(d2, 0.0, out=d2)
                Z = -0.5 * d2 * self._inv_sigma2                       # (B,Nc)

                z_max = np.max(Z, axis=1, keepdims=True)               # (B,1)
                z_max[~np.isfinite(z_max)] = 0.0
                lse = z_max + np.log(np.sum(np.exp(Z - z_max), axis=1, keepdims=True) + 1e-12)  # (B,1)
                lse = lse.ravel()  # (B,)

                m = np.maximum(acc_log[i0:i0 + self.batch], lse)
                acc_log[i0:i0 + self.batch] = m + np.log(
                    np.exp(acc_log[i0:i0 + self.batch] - m) + np.exp(lse - m) + 1e-12
                )

            logs[:, j] = acc_log + self._log_prior[c]

        return logs  # (N, C)

    def predict(self, X):
        logs = self._log_scores(X)
        idx = np.argmax(logs, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        logs = self._log_scores(X)
        m = np.max(logs, axis=1, keepdims=True)
        probs = np.exp(logs - m)
        probs /= (np.sum(probs, axis=1, keepdims=True) + 1e-12)
        return probs


__all__ = ['StablePNN']
