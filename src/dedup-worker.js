/**
 * In-browser embedding worker. Bundled by esbuild into dedup/dedup-worker.bundle.js.
 *
 * Runs inside the OFFSCREEN DOCUMENT, not the page and not a content script. That is
 * forced by X's CSP, measured from live response headers:
 *
 *   worker-src 'self' blob:              -- workers are fine
 *   script-src ... 'wasm-unsafe-eval'    -- WASM is fine
 *   connect-src 'self' https://api.x.com ... (NO huggingface.co)
 *
 * So a worker running on the page origin could start, and then never download the
 * model. An offscreen document runs on the extension's own origin under the extension's
 * CSP, where both the Hugging Face fetch and WebGPU are unconstrained.
 *
 * device:'auto' lets transformers.js order execution providers [webgpu, wasm] when a
 * GPU adapter exists and fall back to [wasm] when it does not -- built into the library,
 * not hand-rolled. Worth knowing on Linux: Chrome's driver bug-list disables Vulkan by
 * default on hybrid-graphics laptops, which silently drops WebGPU to SwiftShader
 * software emulation. That is not a crash, it is a ~200x slowdown that looks like a
 * working GPU (measured: 1339 ms/post emulated vs 2.6 ms/post real).
 */
import { pipeline, env } from '@huggingface/transformers'
import { MODEL } from '../dedup/config.js'

env.allowLocalModels = false

// Point ONNX Runtime at runtime files SHIPPED WITH THE EXTENSION.
//
// Required, not an optimisation. By default ORT loads its backend through a blob:
// dynamic import, and MV3's extension_pages CSP permits only 'self' and
// 'wasm-unsafe-eval' -- blob: cannot be added to it. The result is a hard failure with
// no network activity at all:
//   "no available backend found. ERR: [webgpu] Failed to fetch dynamically imported
//    module: blob:chrome-extension://... , [wasm] previous call to 'initWasm()' failed"
// Serving the .mjs/.wasm from the extension origin removes the blob import entirely.
//
// self.location is dedup/dedup-worker.bundle.js, so './ort/' resolves to dedup/ort/.
env.backends.onnx.wasm.wasmPaths = new URL('./ort/', self.location.href).href
// Single-threaded: cross-origin isolation (COOP/COEP) is not available here, so the
// threaded build cannot spawn its pool and falls back noisily.
env.backends.onnx.wasm.numThreads = 1

let ready

/** transformers.js fires progress_callback per network chunk. For a large weight file
 *  that is thousands of events/sec -- enough to flood the message queue and hang the
 *  host. Time-box it, but always let terminal states through so a progress UI cannot
 *  get stuck at 99%. */
function throttle(post, minMs = 200) {
  let last = 0
  return (d) => {
    const now = Date.now()
    if (d.status === 'done' || d.status === 'ready' || now - last >= minMs) {
      last = now
      post(d)
    }
  }
}

function load() {
  if (!ready) {
    ready = (async () => {
      const progress_callback = throttle((data) =>
        self.postMessage({ type: 'progress', data }))
      try {
        const ex = await pipeline('feature-extraction', MODEL.id, {
          dtype: MODEL.dtype,
          device: 'auto',
          progress_callback,
        })
        // Emit our own readiness signal: transformers.js's per-file events have no
        // single reliable "everything loaded" marker, and having the UI guess from
        // file-level event names is how progress bars end up lying.
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

  // transformers.js implements 'mean' and 'cls' natively. 'lasttoken' is not a
  // supported pooling mode, so it is done by hand from the unpooled output -- three of
  // the eight benchmarked models need it, and silently substituting mean would yield
  // confident nonsense.
  if (MODEL.pooling === 'lasttoken') {
    const out = await ex(input, { pooling: 'none', normalize: false })
    const [b, s, d] = out.dims
    const flat = out.data
    const vecs = []
    for (let i = 0; i < b; i++) {
      const v = new Float32Array(d)
      const off = i * s * d + (s - 1) * d
      for (let k = 0; k < d; k++) v[k] = flat[off + k]
      let n = 0
      for (let k = 0; k < d; k++) n += v[k] * v[k]
      n = Math.sqrt(n) || 1
      for (let k = 0; k < d; k++) v[k] /= n
      vecs.push(v)
    }
    return vecs
  }

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
    self.postMessage({ type: 'result', id: msg.id, vecs },
      vecs.map((v) => v.buffer))
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err?.message || err) })
  }
}
