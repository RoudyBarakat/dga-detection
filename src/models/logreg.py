"""Model 1: logistic regression, written as the four steps of a model.

  1. Define a predictor   predict_proba: P(class | x) = softmax(x W + b)
  2. Define a loss        loss_and_grad: weighted cross-entropy + L2 penalty
  3. Optimize the loss    fit: find W, b with L-BFGS (scipy), from our loss and gradient
  4. Test the model       see src/models/evaluate.py (metrics on validation / test)

One class covers both setups:
  - binary:       classes = ['dga', 'legit']           (softmax with 2 classes =
                                                         ordinary logistic regression)
  - family-first: classes = ['legit', 103 DGA families] (multinomial logistic regression)

Usage:
    model = LogReg(l2=1e-4).fit(X_train, y_train, sample_weight=w_train)
    proba = model.predict_proba(X_val)          # one column per class
"""
import time

import numpy as np
from scipy.optimize import minimize


def softmax(z):
    """Row-wise softmax, computed stably (subtract the row maximum first)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogReg:
    def __init__(self, l2=1e-4, max_iter=500, tol=1e-8):
        self.l2 = l2              # strength of the L2 penalty on the weights
        self.max_iter = max_iter  # maximum number of L-BFGS iterations
        self.tol = tol            # stop when the loss improves less than this

    # ------------------------------------------------------------------ #
    # Preparation: standardize every feature (mean 0, std 1 on train),  #
    # so all features are on the same scale and the optimizer converges. #
    # ------------------------------------------------------------------ #
    def _standardize(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    # ------------------------------------------------------------------ #
    # Step 1: the predictor                                              #
    # ------------------------------------------------------------------ #
    def _scores(self, Xs, W, b):
        """One score per class: z = x W + b (shape: rows x classes)."""
        return Xs @ W + b

    def predict_proba(self, X):
        """P(class | x) = softmax(x W + b), one column per class (self.classes_)."""
        return softmax(self._scores(self._standardize(X), self.W_, self.b_))

    # ------------------------------------------------------------------ #
    # Step 2: the loss                                                   #
    # ------------------------------------------------------------------ #
    def _loss_and_grad(self, params, Xs, Y, w):
        """Weighted cross-entropy + L2 penalty, and its gradient.

        loss = sum_i w_i * (-log P(true class of i)) / sum_i w_i  +  l2/2 * ||W||^2

        Cross-entropy punishes a low probability on the true class; the sample
        weights w_i (from load_splits) make small DGA families count more; the L2
        penalty keeps the weights small (less overfitting). b is not penalized.
        """
        n_feat, n_cls = Xs.shape[1], Y.shape[1]
        W = params[: n_feat * n_cls].reshape(n_feat, n_cls)
        b = params[n_feat * n_cls:]

        z = self._scores(Xs, W, b)
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # log softmax
        w_norm = w / w.sum()

        loss = -(w_norm * (Y * log_p).sum(axis=1)).sum() + self.l2 / 2 * (W ** 2).sum()

        # Gradient: d loss / d z = w_i * (P - Y) / sum w
        dz = (np.exp(log_p) - Y) * w_norm[:, None]
        grad_W = Xs.T @ dz + self.l2 * W
        grad_b = dz.sum(axis=0)
        self._last_loss = loss          # remembered for the loss history (fit)
        return loss, np.concatenate([grad_W.ravel(), grad_b])

    # ------------------------------------------------------------------ #
    # Step 3: optimize the loss                                          #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, sample_weight=None):
        """Learn W and b on the training set.

        X: features (rows x features), y: class of each row (strings),
        sample_weight: one weight per row (None = all equal).
        """
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0          # constant columns: avoid dividing by 0
        Xs = self._standardize(X)

        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        Y = np.zeros((len(y_idx), len(self.classes_)))
        Y[np.arange(len(y_idx)), y_idx] = 1.0     # one-hot: 1 in the true class column
        w = np.ones(len(y_idx)) if sample_weight is None else np.asarray(sample_weight, float)

        n_feat, n_cls = Xs.shape[1], len(self.classes_)
        params0 = np.zeros(n_feat * n_cls + n_cls)   # start from W = 0, b = 0
        self.loss_history_ = []

        result = minimize(self._loss_and_grad, params0, args=(Xs, Y, w), jac=True,
                          method="L-BFGS-B",
                          callback=lambda p: self.loss_history_.append(self._last_loss),
                          options={"maxiter": self.max_iter, "ftol": self.tol})
        self.W_ = result.x[: n_feat * n_cls].reshape(n_feat, n_cls)
        self.b_ = result.x[n_feat * n_cls:]
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.final_loss_ = float(result.fun)
        self.fit_seconds_ = time.time() - start
        return self
