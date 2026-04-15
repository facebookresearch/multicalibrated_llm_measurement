#!/bin/bash
cd "$(dirname "$0")"
pandoc paper.md -o paper.pdf --pdf-engine=pdflatex --citeproc --bibliography=references.bib
pandoc si_appendix.md -o si_appendix.pdf --pdf-engine=pdflatex --citeproc --bibliography=references.bib
