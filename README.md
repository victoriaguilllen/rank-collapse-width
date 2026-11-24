# rank-collapse-width

Proyecto de máster sobre *Mind the Gap: a Spectral Analysis of Rank Collapse and Signal Propagation in Attention Layers* (Nait Saada, Naderi & Tanner, ICML 2025, [arXiv:2410.07799](https://arxiv.org/abs/2410.07799)).

## Entorno

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest
```

El código común está en `sims/attention.py`. Para validar contra el código de los autores:

```bash
git clone https://github.com/thizirinait/Mind-the-Gap-a-Spectral-Analysis-of-Rank-Collapse-and-Signal-Propagation-in-Attention-Layers /ruta/ref
REF_REPO=/ruta/ref .venv/bin/python -m pytest tests/test_referencia.py
```
