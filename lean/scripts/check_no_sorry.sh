#!/usr/bin/env bash
# Track-L acceptance (design.md §3.4): no `sorry`/`admit` anywhere in Layers 0,1,3.
# The deep analytic assumptions live ONLY as honest `structure` hypothesis fields in
# Layer 2 (SffProof/Hypotheses.lean) — that file is exempt from the grep.
set -euo pipefail
cd "$(dirname "$0")/.."

# Match `sorry`/`admit` as code tokens (word boundary), skip the Layer-2 file.
hits=$(grep -rnwE '(sorry|admit)' SffProof SffProof.lean \
        --include='*.lean' \
        | grep -v 'SffProof/Hypotheses.lean' \
        | grep -vE ':[0-9]+:.*(--|/-|no .sorry|sorry. *exist)' \
        || true)

if [ -n "$hits" ]; then
  echo "FAIL: sorry/admit found outside Layer 2:"
  echo "$hits"
  exit 1
fi
echo "OK: no sorry/admit outside SffProof/Hypotheses.lean"
