#!/usr/bin/env bash
# Download GloVe 6B vectors (Wikipedia 2014 + Gigaword 5, 400k uncased tokens).
#
# The archive is ~822 MB and contains the 50d, 100d, 200d and 300d files.
# By default only glove.6B.50d.txt is kept.
#
#   ./scripts/download_glove.sh              # 50d into ./data
#   ./scripts/download_glove.sh 100d ./vecs  # 100d into ./vecs
#   KEEP_ZIP=1 ./scripts/download_glove.sh   # keep the archive
set -euo pipefail

DIM="${1:-50d}"
DEST="${2:-data}"
URL="${GLOVE_URL:-https://nlp.stanford.edu/data/glove.6B.zip}"
MIRROR="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
TARGET="glove.6B.${DIM}.txt"

mkdir -p "$DEST"
if [ -f "$DEST/$TARGET" ]; then
  echo "$DEST/$TARGET already present; nothing to do."
  exit 0
fi

ZIP="$DEST/glove.6B.zip"
if [ ! -f "$ZIP" ]; then
  echo "Downloading $URL (~822 MB) ..."
  curl -fL --retry 4 --retry-delay 2 -o "$ZIP.part" "$URL" \
    || { echo "Primary host failed, trying $MIRROR ..." >&2
         curl -fL --retry 4 --retry-delay 2 -o "$ZIP.part" "$MIRROR"; }
  mv "$ZIP.part" "$ZIP"
fi

echo "Extracting $TARGET ..."
unzip -o "$ZIP" "$TARGET" -d "$DEST"
[ -n "${KEEP_ZIP:-}" ] || rm -f "$ZIP"

echo "Ready: $DEST/$TARGET"
echo "Try:   python -m glove_retrieval --vectors $DEST/$TARGET --text king -k 10"
