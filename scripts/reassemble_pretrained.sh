#!/usr/bin/env bash
set -euo pipefail

# Reassemble the provided split checkpoint parts into a single .ckpt file.

PART_PREFIX="pretrained/gisc_fmt_brain1000_best.ckpt.part"
OUT="pretrained/gisc_fmt_brain1000_best.ckpt"
SUMS="pretrained/SHA256SUMS"

if [[ -f "${OUT}" ]]; then
  echo "[OK] ${OUT} already exists"
  exit 0
fi

parts=( ${PART_PREFIX}* )
if [[ ${#parts[@]} -eq 0 ]]; then
  echo "[ERROR] No parts found: ${PART_PREFIX}*" >&2
  exit 1
fi

cat "${PART_PREFIX}"* > "${OUT}"

echo "[OK] Reassembled: ${OUT}"

if [[ -f "${SUMS}" ]]; then
  (cd pretrained && sha256sum -c SHA256SUMS)
fi
