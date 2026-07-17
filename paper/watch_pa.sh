#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# Continuous build for the Political Analysis version. Rebuilds
# paper_pa.pdf / si_appendix_pa.pdf whenever the matching .md (or
# references.bib) changes. Zero dependencies: mtime polling.
#
# Usage:  bash watch_pa.sh      (Ctrl-C to stop)

cd "$(dirname "$0")"

build() {  # $1 = source .md, $2 = output .pdf
  printf '[%s] building %s ... ' "$(date +%H:%M:%S)" "$2"
  if pandoc "$1" -o "$2" --pdf-engine=pdflatex --citeproc \
      --bibliography=references.bib --csl=chicago-author-date.csl 2>/tmp/pandoc_pa.err; then
    echo "ok"
  else
    echo "FAILED"
    sed 's/^/    /' /tmp/pandoc_pa.err
  fi
}

mtime() { stat -f %m "$1" 2>/dev/null || echo 0; }

echo "Watching paper_pa.md, si_appendix_pa.md, references.bib (Ctrl-C to stop)"
p=$(mtime paper_pa.md); s=$(mtime si_appendix_pa.md); b=$(mtime references.bib)
while true; do
  sleep 1
  np=$(mtime paper_pa.md); ns=$(mtime si_appendix_pa.md); nb=$(mtime references.bib)
  [ "$np" != "$p" ] || [ "$nb" != "$b" ] && build paper_pa.md      paper_pa.pdf
  [ "$ns" != "$s" ] || [ "$nb" != "$b" ] && build si_appendix_pa.md si_appendix_pa.pdf
  p=$np; s=$ns; b=$nb
done
