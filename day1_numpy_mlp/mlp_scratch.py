"""
Day 1: NumPy MLP (binary classification)

Fill the TODOs. Do not import deep learning frameworks.
"""
import csv, math, os
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Utils ------------------
def load_csv(path):
    xs, ys = [], []
    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            x1, x2, y = line.strip().split(",")
            xs.append([float(x1), float(x2)])
            ys.append(int(y))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def sigmoid(z):
    # TODO: implement numerically stable sigmoid
    raise NotImplementedError

def bce_loss(logits, targets):
    """
    logits: raw scores BEFORE sigmoid, shape (N, 1)
    targets: 0/1, shape (N, 1)
    Return: scalar loss
    """
    # TODO: implement BCE using logits + sigmoid
    raise NotImplementedError

def accuracy(logits, targets):
    # TODO: compute accuracy given logits (pre-sigmoid)
    raise NotImplementedError

# ------------------ Model ------------------
class MLP:
    def __init__(self, in_dim=2, hidden=16):
        rng = np.random.default_rng(0)
        # He initialization for ReLU
        self.W1 = rng.normal(0, math.sqrt(2/in_dim), size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = rng.normal(0, math.sqrt(2/hidden), size=(hidden, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

    def forward(self, x):
        """
        Return intermediates for backprop.
        """
        # TODO: affine1 -> ReLU -> affine2
        raise NotImplementedError

    def backward(self, cache, targets):
        """
        cache: dict from forward with intermediates
        targets: (N,1)
        Returns grads: dW1, db1, dW2, db2
        """
        # TODO: backprop for BCE-with-logits
        raise NotImplementedError

    def step(self, grads, lr):
        # TODO: SGD update
        raise NotImplementedError

# ------------------ Training ------------------
def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    idx = np.arange(len(y))
    if shuffle: np.random.shuffle(idx)
    for start in range(0, len(y), batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        xb = X[batch_idx]
        yb = y[batch_idx].reshape(-1,1)
        yield xb, yb

def plot_learning_curve(losses, accs):
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(accs)
    ax2.set_ylabel("Val Acc")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.show()

def main():
    Xtr, ytr = load_csv("train.csv")
    Xva, yva = load_csv("val.csv")

    model = MLP(in_dim=2, hidden=16)
    lr = 0.05
    epochs = 200
    losses, val_accs = [], []

    for epoch in range(epochs):
        for xb, yb in iterate_minibatches(Xtr, ytr, batch_size=64, shuffle=True):
            # 1) forward
            # 2) compute loss
            # 3) backward
            # 4) step
            # TODO: fill training steps
            pass

        # Evaluate on val at end of epoch
        # TODO: compute val accuracy
        val_acc = 0.0
        val_accs.append(val_acc)

    # TODO: plot_learning_curve(losses, val_accs)

    # TODO (bonus): draw decision boundary over a grid

if __name__ == "__main__":
    main()
