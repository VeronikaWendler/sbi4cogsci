#!/usr/bin/env bash
# Rebuild tutorial notebooks from their `_`-prefixed percent-format sources.
#
#   ./build_notebooks.sh                      # rebuild all
#   ./build_notebooks.sh day2_bayes_toolkit/_pymc-intro.py   # rebuild one
#
# Source of truth is `_<name>.py` (readable, diffable, and skipped by Quarto
# because of the leading underscore). The committed artifact is `<name>.ipynb`
# WITH stored outputs, because the Pages CI has no Python and never executes
# anything (tutorials/_metadata.yml sets `execute: enabled: false`).
#
# The stems deliberately differ (`_pymc-intro` vs `pymc-intro`): Quarto
# silently skips an .ipynb whose stem matches a sibling .py, which would make
# the page vanish from the site with no error.
set -euo pipefail
cd "$(dirname "$0")"

sources=("$@")
if [ ${#sources[@]} -eq 0 ]; then
  # bash 3.2 (macOS default) has no readarray
  sources=()
  while IFS= read -r line; do sources+=("$line"); done \
    < <(find . -name '_*.py' -not -path './.venv/*' | sort)
fi

for src in "${sources[@]}"; do
  out="$(dirname "$src")/$(basename "$src" .py | sed 's/^_//').ipynb"
  echo "==> $src  ->  $out"
  uv run --with jupytext --with nbconvert --with ipykernel \
    jupytext --to ipynb "$src" -o "$out" --quiet
  uv run --with jupytext --with nbconvert --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 "$out"
  echo "    done: $(du -h "$out" | cut -f1)"
done
