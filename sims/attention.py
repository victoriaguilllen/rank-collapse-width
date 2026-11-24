"""Capa de atención aleatoria en inicialización y utilidades espectrales.

Notación (como en el paper): X ∈ R^{n×d} son n tokens de dimensión d,
A ∈ R^{n×n} es la matriz de atención (estocástica por filas).
"""

import numpy as np


def random_attention_matrix(
    X,
    d_k=None,
    *,
    sigma=1.0,
    temperature=1.0,
    rng=None,
    W_Q=None,
    W_K=None,
):
    """Matriz de atención A = softmax(X W_Q W_Kᵀ Xᵀ / (√d_k · temperature)).

    Si no se pasan, W_Q y W_K ∈ R^{d×d_k} se muestrean con entradas
    i.i.d. N(0, σ²), como en la ec. (1) del paper (σ = σ_qk). Con entrada
    isométrica (X Xᵀ = I) los logits son O(1) y se cumple el Teorema 1:
    σ₁(A) → 1 y √n·σ₂(A) → 2σ.

    Ojo con la escala de X: con filas de norma ≈ √d (p. ej. X gaussiana) hay
    que usar sigma ≈ 1/√d (la `xavier_normal` del repo de los autores) para
    que los logits sigan siendo O(1).

    Parámetros
    ----------
    X : array (n, d)
    d_k : dimensión de queries/keys (por defecto d)
    sigma : desviación típica de las entradas de W_Q y W_K (σ_qk)
    temperature : divide los logits; >1 suaviza A, <1 la concentra
    rng : np.random.Generator o semilla, para reproducibilidad
    W_Q, W_K : pesos fijos (d, d_k); si se dan, se ignoran sigma y rng

    Devuelve
    --------
    A : array (n, n), entradas positivas y filas que suman 1
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz (n, d); tiene forma {X.shape}")
    d = X.shape[1]
    if d_k is None:
        d_k = W_Q.shape[1] if W_Q is not None else d

    rng = np.random.default_rng(rng)
    if W_Q is None:
        W_Q = rng.normal(0.0, sigma, size=(d, d_k))
    if W_K is None:
        W_K = rng.normal(0.0, sigma, size=(d, d_k))

    logits = (X @ W_Q) @ (X @ W_K).T / (np.sqrt(d_k) * temperature)
    logits -= logits.max(axis=1, keepdims=True)  # softmax numéricamente estable
    A = np.exp(logits)
    A /= A.sum(axis=1, keepdims=True)
    return A


def stable_rank(X):
    """Stable rank ‖X‖²_F / ‖X‖²₂ = Σ σᵢ² / σ₁².

    Está entre 1 (rango 1) y rank(X). Devuelve 0.0 para la matriz nula.
    """
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0.0
    return float(np.sum(s**2) / s[0] ** 2)
