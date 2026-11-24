"""Validación contra el código de los autores.

Repo: https://github.com/thizirinait/Mind-the-Gap-a-Spectral-Analysis-of-Rank-Collapse-and-Signal-Propagation-in-Attention-Layers

Se ejecuta solo si REF_REPO apunta a un clon local de ese repo y torch está
instalado:

    git clone <url> /ruta/ref
    REF_REPO=/ruta/ref pytest tests/test_referencia.py
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from attention import random_attention_matrix, stable_rank

torch = pytest.importorskip("torch")
REF_REPO = os.environ.get("REF_REPO")
pytestmark = pytest.mark.skipif(not REF_REPO, reason="REF_REPO no definido")


def _cargar_atencion_referencia():
    ruta = Path(REF_REPO) / "atttention" / "ScaledDotProductAttention.py"
    spec = importlib.util.spec_from_file_location("ref_sdpa", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScaledDotProductAttention


@pytest.mark.parametrize("n,d", [(16, 16), (64, 32), (128, 256)])
def test_misma_matriz_de_atencion(n, d):
    rng = np.random.default_rng(n + d)
    X = rng.normal(size=(n, d))

    Ref = _cargar_atencion_referencia()
    ref = Ref(d, d, distribution_Q="xavier_normal", distribution_K="xavier_normal")
    ref = ref.double()
    # nn.Linear calcula x Wᵀ, así que nuestro W_Q es su weight traspuesto
    W_Q = ref.query_embed.weight.detach().numpy().T
    W_K = ref.key_embed.weight.detach().numpy().T

    Xt = torch.from_numpy(X)
    with torch.no_grad():
        A_ref = ref.get_attention_weights(ref.query_embed(Xt), ref.key_embed(Xt))

    A = random_attention_matrix(X, W_Q=W_Q, W_K=W_K)
    np.testing.assert_allclose(A, A_ref.numpy(), rtol=1e-10, atol=1e-14)


def test_misma_escala_de_pesos_que_xavier():
    d = 256
    Ref = _cargar_atencion_referencia()
    torch.manual_seed(0)
    ref = Ref(d, d, distribution_Q="xavier_normal", distribution_K="xavier_normal")
    std_ref = ref.query_embed.weight.detach().numpy().std()
    W = np.random.default_rng(0).normal(0.0, 1 / np.sqrt(d), size=(d, d))
    assert std_ref == pytest.approx(W.std(), rel=0.02)


def test_stable_rank_de_la_referencia_es_el_de_X_Xt():
    # stable_ranks.py calcula Σ s²/max s² con s = svdvals(X Xᵀ), es decir,
    # el stable rank de X Xᵀ (Σσᵢ⁴/σ₁⁴), no el de X (Σσᵢ²/σ₁²).
    X = np.random.default_rng(1).normal(size=(40, 20))
    Xt = torch.from_numpy(X)
    s = torch.linalg.svdvals(Xt @ Xt.T)
    sr_ref = float((s**2).sum() / (s**2).max())
    assert stable_rank(X @ X.T) == pytest.approx(sr_ref)
    assert stable_rank(X) != pytest.approx(sr_ref)
