
LYX:=lyx
PDFLATEX:=pdflatex
BIBTEX:=bibtex

all: hypot.pdf

%.pdf: %.tex
	$(PDFLATEX) $* && \
	$(BIBTEX) $* && \
	$(PDFLATEX) $* && \
	$(PDFLATEX) $*

%.tex: %.lyx
	$(LYX) --export-to pdflatex $@ $<

hypot.pdf: figs/*.pdf

clean:
	-rm -rf *.log *.aux *.out *.bbl *.blg
