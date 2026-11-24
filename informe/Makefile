.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf informe.pdf

clean:
	latexmk -C -outdir=build
	rm -rf build
