import numpy as np
import pytest

from attention import random_attention_matrix, stable_rank


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(64, 32))


def test_attention_es_estocastica_por_filas(X):
    A = random_attention_matrix(X, rng=1)
    n = X.shape[0]
    assert A.shape == (n, n)
    assert np.all(A > 0)
    np.testing.assert_allclose(A.sum(axis=1), 1.0)
    # 𝟙 es autovector con autovalor 1, luego σ₁(A) ≥ 1
    np.testing.assert_allclose(A @ np.ones(n), np.ones(n))
    assert np.linalg.norm(A, 2) >= 1 - 1e-12


def test_attention_reproducible_con_semilla(X):
    np.testing.assert_array_equal(
        random_attention_matrix(X, rng=7), random_attention_matrix(X, rng=7)
    )
    assert not np.allclose(
        random_attention_matrix(X, rng=7), random_attention_matrix(X, rng=8)
    )


def test_attention_con_pesos_fijos(X):
    d, d_k = X.shape[1], 8
    rng = np.random.default_rng(3)
    W_Q, W_K = rng.normal(size=(d, d_k)), rng.normal(size=(d, d_k))
    logits = X @ W_Q @ W_K.T @ X.T / np.sqrt(d_k)
    esperado = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        random_attention_matrix(X, W_Q=W_Q, W_K=W_K), esperado, rtol=1e-10
    )


def test_softmax_estable_con_logits_grandes(X):
    A = random_attention_matrix(1e3 * X, rng=0)
    assert np.all(np.isfinite(A))
    np.testing.assert_allclose(A.sum(axis=1), 1.0)


def test_temperatura(X):
    # temperatura muy alta → A ≈ (1/n) 𝟙𝟙ᵀ
    A = random_attention_matrix(X, rng=0, temperature=1e12)
    np.testing.assert_allclose(A, 1 / X.shape[0], rtol=1e-6)


@pytest.mark.parametrize("sigma,d_k", [(1.0, None), (2.0, 8)])
def test_pesos_por_defecto_son_normales_sigma(X, sigma, d_k):
    # Los pesos por defecto deben ser W_Q, W_K ~ N(0, σ²), muestreados en ese orden.
    d = X.shape[1]
    k = d if d_k is None else d_k
    rng = np.random.default_rng(5)
    W_Q = rng.normal(0.0, sigma, size=(d, k))
    W_K = rng.normal(0.0, sigma, size=(d, k))
    np.testing.assert_allclose(
        random_attention_matrix(X, d_k, sigma=sigma, rng=5),
        random_attention_matrix(X, W_Q=W_Q, W_K=W_K),
    )


def test_stable_rank_casos_conocidos():
    assert stable_rank(np.eye(10)) == pytest.approx(10)
    u, v = np.arange(1, 6.0), np.arange(1, 4.0)
    assert stable_rank(np.outer(u, v)) == pytest.approx(1)
    assert stable_rank(np.diag([2.0, 1.0])) == pytest.approx(1.25)
    assert stable_rank(np.zeros((3, 3))) == 0.0


def test_stable_rank_acotado(X):
    r = stable_rank(X)
    assert 1 <= r <= np.linalg.matrix_rank(X)


def test_stable_rank_matriz_uniforme_es_uno():
    n = 50
    assert stable_rank(np.full((n, n), 1 / n)) == pytest.approx(1)


def test_teorema_1_entrada_isometrica():
    # X Xᵀ = I y pesos N(0, 1): σ₁(A) ≈ 1 y √n·σ₂(A) = O(1), sin depender de n
    escalados = []
    for n in (64, 256):
        rng = np.random.default_rng(n)
        X = np.linalg.qr(rng.normal(size=(n, n)))[0]
        s = np.linalg.svd(random_attention_matrix(X, rng=rng), compute_uv=False)
        assert s[0] == pytest.approx(1, abs=0.05)
        escalados.append(np.sqrt(n) * s[1])
    assert escalados[1] == pytest.approx(escalados[0], rel=0.25)
