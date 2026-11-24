# rank-collapse-width
## Informe

En `informe/`: `main.tex` incluye una sección por archivo en `informe/secciones/`, con la notación común en `notacion.tex` y la bibliografía en `refs.bib` (biblatex + biber). Las figuras van en `figuras/` (en la raíz) y se incluyen sin ruta, p. ej. `\includegraphics{exp1_gap_vs_n}`.

```bash
make -C informe pdf   # genera informe/informe.pdf
```

El texto pendiente se marca con `\pendiente{...}` y sale en naranja en el PDF.
