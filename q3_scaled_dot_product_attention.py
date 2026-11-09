"""
Part II, Q3: Scaled dot-product attention with NumPy.
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
"""
import numpy as np


def softmax(x):
    """Row-wise softmax (subtract the row max for numerical stability)."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d = K.shape[-1]                      # key dimension
    scores = Q @ K.T                     # 1. dot product of Q and K^T
    scaled = scores / np.sqrt(d)         # 2. scale by sqrt(d)
    weights = softmax(scaled)            # 3. softmax -> attention weights
    output = weights @ V                 # 4. weighted sum of the values
    return weights, output


if __name__ == "__main__":
    Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    weights, output = scaled_dot_product_attention(Q, K, V)
    np.set_printoptions(precision=4, suppress=True)
    print("Attention weights (after softmax):\n", weights)
    print("\nFinal output matrix:\n", output)
