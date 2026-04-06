from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier


class FingerprintNet:
    def __init__(self, classifier: MLPClassifier, window_size: int, num_features: int):
        self.classifier = classifier
        self.window_size = window_size
        self.num_features = num_features

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), self.window_size * self.num_features)

    def embed(self, X: np.ndarray) -> np.ndarray:
        activations = self._flatten(X)
        hidden_count = len(self.classifier.coefs_) - 1
        for layer_idx in range(hidden_count):
            activations = activations @ self.classifier.coefs_[layer_idx] + self.classifier.intercepts_[layer_idx]
            activations = np.maximum(activations, 0.0)
        return activations


def train_fingerprint_model(
    X: np.ndarray,
    y: np.ndarray,
    num_players: int,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[FingerprintNet, dict[str, float]]:
    flat_X = X.reshape(len(X), X.shape[1] * X.shape[2])
    classifier = MLPClassifier(
        hidden_layer_sizes=(96, 32),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=epochs,
        random_state=7,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    classifier.fit(flat_X, y)
    model = FingerprintNet(classifier=classifier, window_size=X.shape[1], num_features=X.shape[2])
    train_acc = float(classifier.score(flat_X, y))
    loss_curve = classifier.loss_curve_[-1] if classifier.loss_curve_ else 0.0
    return model, {"train_loss": float(loss_curve), "train_accuracy": train_acc}


def embed_windows(model: FingerprintNet, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    embeddings = []
    for idx in range(0, len(X), batch_size):
        batch = X[idx : idx + batch_size]
        embeddings.append(model.embed(batch))
    return np.concatenate(embeddings, axis=0)
