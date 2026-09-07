#!/bin/bash
# Assemble a clean, loadable extension directory.
#
# Required, not cosmetic: the repo root holds node_modules (824M), a Python .venv
# (531M) and the benchmark corpus (214M), and Chrome scans EVERY file under a
# --load-extension directory. Pointing it at the repo root does not error usefully,
# it just fails to load.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist-dedup
rm -rf "$OUT"; mkdir -p "$OUT"

npm run build:worker >/dev/null

cp manifest.mv3.json "$OUT/manifest.json"
cp background.js content.js script.js dedup.css offscreen.html offscreen.js "$OUT/"
cp options.html options.js options.css options-icon.png "$OUT/" 2>/dev/null || true
[ -f browser_action.html ] && cp browser_action.html "$OUT/"
cp -r icons _locales "$OUT/"
mkdir -p "$OUT/dedup"
cp dedup/*.js "$OUT/dedup/"

# Ship ONNX Runtime's own wasm/mjs from the extension origin. Without these ORT tries a
# blob: dynamic import, which MV3's extension_pages CSP forbids (only 'self' and
# 'wasm-unsafe-eval' are permitted there), and every backend fails to initialise.
mkdir -p "$OUT/dedup/ort"
# ORT ships four wasm variants (plain / jsep / asyncify / jspi) and picks by feature
# detection, so this started out copying all four -- 74MB -- rather than guessing wrong
# and getting a bare "no available backend found". The set is now pinned to what the
# built worker can actually ask for, read out of the bundle itself:
#
#   grep -o 'ort-wasm-simd-threaded[a-z.]*\.wasm' dist-dedup/dedup/dedup-worker.bundle.js
#     -> ort-wasm-simd-threaded.asyncify.wasm   (WebGPU: transformers.js pulls
#                                                onnxruntime-web/webgpu = ort.webgpu.bundle)
#     -> ort-wasm-simd-threaded.wasm            (CPU fallback: onnxruntime-web/wasm)
#
# jsep (26MB) and jspi (15MB) are named only by ORT entry points this build never
# imports, so shipping them cost 41MB for nothing. Re-run that grep after any bump of
# onnxruntime-web or @huggingface/transformers -- a changed entry point changes the set.
for v in ort-wasm-simd-threaded.asyncify ort-wasm-simd-threaded; do
  cp "node_modules/onnxruntime-web/dist/$v.wasm" "$OUT/dedup/ort/"
  cp "node_modules/onnxruntime-web/dist/$v.mjs"  "$OUT/dedup/ort/"
done
echo "  ort runtime: $(du -sh "$OUT/dedup/ort" | cut -f1)"

echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
find "$OUT" -maxdepth 2 -type f | sed "s|^$OUT/|  |" | sort | head -30
