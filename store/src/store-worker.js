/**
 * In-browser embedding worker for the standalone extension. Bundled by esbuild into
 * dedup/dedup-worker.bundle.js.
 *
 * Runs inside the OFFSCREEN DOCUMENT, not the page and not a content script: a worker
 * created from a content script belongs to the x.com origin and is governed by that
 * page's CSP, while here the extension's own CSP applies.
 *
 * The model is BUNDLED, not downloaded. That is a deliberate trade -- it makes the
 * package ~190MB -- and it buys three things:
 *   * no host permissions at all, so the extension has no network surface to justify;
 *   * no question about Chrome's remotely-hosted-code policy, since nothing is fetched;
 *   * it works offline and on first paint, with no download-progress UI to get wrong.
 *
 * device:'auto' lets transformers.js order execution providers [webgpu, wasm] when a GPU
 * adapter exists and fall back to [wasm] when it does not. Worth knowing on Linux:
 * Chrome's driver bug-list disables Vulkan by default on hybrid-graphics laptops, which
 * silently drops WebGPU to SwiftShader software emulation -- not a crash, a ~200x
 * slowdown that looks like a working GPU (measured: 1339 ms/post emulated vs 2.6 real).
 */
import { pipeline, env } from '@huggingface/transformers'
import { MODEL } from '../dedup/config.js'

// Local only. allowRemoteModels=false is the load-bearing line: without it a missing or
// misnamed local file silently falls back to a Hugging Face fetch, which would fail at
// runtime with no host permission and, worse, would mean the shipped extension did not
// actually behave the way its store listing claims.
env.allowLocalModels = true
env.allowRemoteModels = false
// self.location is dedup/dedup-worker.bundle.js, so '../models/' is <root>/models/.
env.localModelPath = new URL('../models/', self.location.href).href
env.useBrowserCache = false          // the files are already local; caching them again wastes quota

// Point ONNX Runtime at runtime files SHIPPED WITH THE EXTENSION.
//
// Required, not an optimisation. By default ORT loads its backend through a blob:
// dynamic import, and MV3's extension_pages CSP permits only 'self' and
// 'wasm-unsafe-eval' -- blob: cannot be added to it. The result is a hard failure with
// no network activity at all: "no available backend found ... Failed to fetch
// dynamically imported module: blob:chrome-extension://...".
env.backends.onnx.wasm.wasmPaths = new URL('./ort/', self.location.href).href
// Single-threaded: cross-origin isolation (COOP/COEP) is not available here, so the
// threaded build cannot spawn its pool and falls back noisily.
env.backends.onnx.wasm.numThreads = 1

/** Folder name under models/, NOT the Hugging Face repo id -- with allowRemoteModels
 *  off, this string is resolved only against env.localModelPath. */
const LOCAL_ID = 'bekko-a8m'

let ready

function load() {
  if (!ready) {
    ready = (async () => {
      try {
        const ex = await pipeline('feature-extraction', LOCAL_ID, {
          dtype: MODEL.dtype,
          device: 'auto',
        })
        self.postMessage({ type: 'progress', data: { status: 'ready' } })
        return ex
      } catch (err) {
        ready = undefined      // allow a retry on the next call
        throw err
      }
    })()
  }
  return ready
}

async function embed(texts) {
  const ex = await load()
  const input = MODEL.prompt ? texts.map((t) => MODEL.prompt + t) : texts
  const out = await ex(input, { pooling: MODEL.pooling, normalize: true })
  const [b, d] = out.dims
  const vecs = []
  for (let i = 0; i < b; i++) vecs.push(new Float32Array(out.data.slice(i * d, (i + 1) * d)))
  return vecs
}

self.onmessage = async (ev) => {
  const msg = ev.data
  if (msg?.type !== 'embed') return
  try {
    const vecs = await embed(msg.texts)
    // Transfer the buffers rather than structured-cloning them; at 384 floats per post
    // and 32 posts per batch this is the difference between a copy and a pointer move.
    self.postMessage({ type: 'result', id: msg.id, vecs }, vecs.map((v) => v.buffer))
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) })
  }
}
