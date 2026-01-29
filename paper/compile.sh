#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

cd "$(dirname "$0")"
pandoc paper.md -o paper.pdf --pdf-engine=pdflatex --citeproc --bibliography=references.bib
pandoc si_appendix.md -o si_appendix.pdf --pdf-engine=pdflatex --citeproc --bibliography=references.bib
