#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# Builds the Political Analysis version (paper_pa.md + si_appendix_pa.md).
# The original PNAS version is still built by compile.sh.

cd "$(dirname "$0")"
pandoc paper_pa.md -o paper_pa.pdf --pdf-engine=pdflatex --citeproc \
  --bibliography=references.bib --csl=chicago-author-date.csl
pandoc si_appendix_pa.md -o si_appendix_pa.pdf --pdf-engine=pdflatex --citeproc \
  --bibliography=references.bib --csl=chicago-author-date.csl
