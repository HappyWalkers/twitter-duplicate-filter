#!/usr/bin/env bash
# Build the standalone Chrome Web Store package into dist-store/ and zip it.
#
# This is NOT the CPFT fork. It ships the deduplication feature alone, under its own
# name and icons, and coordinates with Control Panel for Twitter through CSS classes
# only -- so a user can install both and they cooperate rather than collide.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dist-store
rm -rf "$OUT"
mkdir -p "$OUT"

# Extension sources
cp store/manifest.json store/content.js store/background.js \
   store/offscreen.html store/offscreen.js \
   store/popup.html store/popup.js "$OUT/"
# From the repo root, not a copy under store/. There used to be a store/dedup.css and it
# went stale exactly as store/dedup/*.js had: a rule added to the root file never reached
# the build, so a class was applied to elements that no stylesheet ever hid.
cp dedup.css "$OUT/"
cp -r store/icons "$OUT/"
# ONE source of truth for the dedup modules. store/ previously kept its own copy and the
# two silently diverged (the threshold setter existed in one, the stats change in the
# other); copying from dedup/ at build time makes that impossible.
mkdir -p "$OUT/dedup"
cp dedup/*.js "$OUT/dedup/"

# The bundled model. This is the bulk of the package and the reason it needs no host
# permissions: nothing is fetched at runtime.
cp -r store/models "$OUT/"

# Inference worker
npx --yes esbuild store/src/store-worker.js \
  --bundle --format=esm --platform=browser \
  --outfile="$OUT/dedup/dedup-worker.bundle.js" --log-level=warning

# ONNX Runtime wasm, pinned to what the built worker can actually request:
#   grep -o 'ort-wasm-simd-threaded[a-z.]*\.wasm' dist-store/dedup/dedup-worker.bundle.js
# -> asyncify (WebGPU) and plain (CPU fallback). jsep and jspi are named only by ORT
# entry points this build never imports; shipping them cost 41MB for nothing.
mkdir -p "$OUT/dedup/ort"
for v in ort-wasm-simd-threaded.asyncify ort-wasm-simd-threaded; do
  cp "node_modules/onnxruntime-web/dist/$v.wasm" "$OUT/dedup/ort/"
  cp "node_modules/onnxruntime-web/dist/$v.mjs"  "$OUT/dedup/ort/"
done

# Fail loudly if the bundle asks for a variant we did not ship: the runtime error is a
# bare "no available backend found", which is a miserable thing to debug post-release.
missing=0
for w in $(grep -o 'ort-wasm-simd-threaded[a-z.]*\.wasm' "$OUT/dedup/dedup-worker.bundle.js" | sort -u); do
  [ -f "$OUT/dedup/ort/$w" ] || { echo "MISSING ORT variant: $w"; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

# Same check for the model: allowRemoteModels is off, so a missing file is a hard
# failure at first embed rather than a silent download.
for f in config.json tokenizer.json tokenizer_config.json onnx/model.onnx; do
  [ -f "$OUT/models/bekko-a8m/$f" ] || { echo "MISSING model file: $f"; exit 1; }
done

# Every CSS class the code applies must have a rule in the stylesheet. This exact failure
# shipped once: .CpftDupSeen was added to elements while its display:none rule lived only
# in a file the build did not copy, so posts were marked collapsed and stayed fully
# visible. A missing rule is silent -- the class applies, nothing happens, and the symptom
# looks like broken logic rather than a missing style.
missing=0
for cls in $(grep -ohE "classList\.(add|toggle)\('(Cpft[A-Za-z]+)'" "$OUT"/dedup/*.js \
             | grep -oE "Cpft[A-Za-z]+" | sort -u); do
  grep -qE "\.$cls([^A-Za-z0-9_-]|$)" "$OUT/dedup.css" \
    || { echo "CSS RULE MISSING for class: $cls"; missing=1; }
done
# Classes created via className= too.
for cls in $(grep -ohE "className = '([^']*)'" "$OUT"/dedup/*.js | grep -oE "Cpft[A-Za-z]+" | sort -u); do
  grep -qE "\.$cls([^A-Za-z0-9_-]|$)" "$OUT/dedup.css" \
    || { echo "CSS RULE MISSING for class: $cls"; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

( cd "$OUT" && zip -qr ../dist-store.zip . )
echo "built $OUT ($(du -sh "$OUT" | cut -f1)) -> dist-store.zip ($(du -h ../x-timeline-dedup/dist-store.zip 2>/dev/null | cut -f1 || du -h dist-store.zip 2>/dev/null | cut -f1))"
du -sh "$OUT"/* | sort -rh
