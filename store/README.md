# Timeline Dedup for X — standalone extension

The deduplication feature packaged on its own for the Chrome Web Store, with no Control
Panel for Twitter code in it. The two are designed to run together: CPFT owns the
`.HiddenTweet` class, this owns `.CpftDup`, CSS unions them, and neither reads the
other's bookkeeping.

## Build

```bash
./scripts/build-store.sh          # -> dist-store/ and dist-store.zip
```

The model is **not** in git — 157MB of weights and vocabulary exceed GitHub's file limit.
Fetch it before the first build:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
import shutil, pathlib
src = pathlib.Path(snapshot_download("hotchpotch/bekko-embedding-v1-a8m"))
dst = pathlib.Path("store/models/bekko-a8m"); (dst / "onnx").mkdir(parents=True, exist_ok=True)
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    if (src / f).exists(): shutil.copy(src / f, dst / f)
shutil.copy(src / "onnx" / "model.onnx", dst / "onnx" / "model.onnx")
PY
```

`build-store.sh` fails loudly if either the model files or the ONNX Runtime variants the
bundle actually requests are missing, rather than producing a package that installs fine
and then cannot embed anything.

## Why the model is bundled rather than downloaded

It makes the package ~116MB zipped, which is the cost. What it buys:

* **no host permissions at all** — nothing to justify in review, and the extension is
  incapable of contacting any server;
* **no exposure to Chrome's remotely-hosted-code policy**, which is the single most
  likely cause of a rejection for an extension that ships a model;
* it works offline, on first paint, with no download-progress UI to get wrong.

`env.allowRemoteModels = false` in the worker enforces this: if a packaged file were
missing, transformers.js would otherwise fall back to a Hugging Face fetch, and the
shipped extension would quietly stop matching its own privacy claims.

## Submitting

See `STORE-LISTING.md` for the listing copy, permission justifications and asset
checklist, and `PRIVACY.md` for the policy the listing must link to.
