#!/usr/bin/env python3
"""
Drive data/speed/index.html against an ALREADY-RUNNING, GPU-enabled Chrome over CDP.

Why not agy's run.mjs: it launches its own headless Chrome, which on this machine gets
no WebGPU adapter and silently falls back to SwiftShader software emulation. That is how
gte-multilingual-base "measured" 1339 ms/post on nominal WebGPU -- a number that says
nothing about the real extension. requestAdapter() returns null in headless AND in
default headed Chrome here, because Chrome's driver bug-list disables Vulkan on this
hybrid-graphics laptop; launching with --enable-features=Vulkan fixes it (documented in
the sibling bilibili project, confirmed again here: vendor=nvidia, architecture=blackwell).

Start Chrome first:
  google-chrome --user-data-dir=~/.cache/x-dedup-profile --remote-debugging-port=9222 \
      --enable-features=Vulkan --enable-unsafe-webgpu
"""
import argparse, asyncio, functools, http.server, json, socket, threading
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).parent
CDP = "http://127.0.0.1:9222"

# q8-vs-fp32 on WebGPU: the first run showed q8 models at 116-509 ms/post and fp32
# models at 1.9-3.9 ms/post. If that is a dtype effect rather than a model effect,
# the SAME model should get much faster at fp32. Testing both dtypes on gte and mmini
# isolates it.
MODELS = [
    ("onnx-community/gte-multilingual-base", "fp32"),
    ("Xenova/paraphrase-multilingual-MiniLM-L12-v2", "fp32"),
    ("hotchpotch/bekko-embedding-v1-a8m", "fp32"),
]


def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="webgpu")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    texts = [json.loads(l)["text"] for l in
             (HERE.parent / "posts.jsonl").read_text().split("\n") if l.strip()]
    texts = [t for t in texts if len(t) >= 40][:20]

    httpd, port = serve(HERE)
    results = []
    async with async_playwright() as pw:
        b = await pw.chromium.connect_over_cdp(CDP)
        ctx = b.contexts[0]
        for model, dtype in MODELS:
            page = await ctx.new_page()
            try:
                await page.goto(f"http://127.0.0.1:{port}/index.html",
                                wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_function("window.BENCH_READY === true", timeout=120000)
                adapter = await page.evaluate(
                    "async()=>{const a=await navigator.gpu?.requestAdapter?.();"
                    "return a? (a.info?.vendor||'gpu') : 'NONE'}")
                print(f"\n>>> {model} [{dtype}] on {args.device} (adapter={adapter})", flush=True)
                r = await page.evaluate(
                    "async ([m,d,dev,t]) => await window.runBench(m,d,dev,t)",
                    [model, dtype, args.device, texts])
                r["adapter"] = adapter
                results.append(r)
                if r.get("ok"):
                    print(f"    load {r['loadMs']/1000:.1f}s | {r['warmMsPerPost']:.1f} ms/post "
                          f"| dim={r['dim']} | actualDevice={r.get('actualDevice')}", flush=True)
                else:
                    print(f"    FAILED: {r.get('error')}", flush=True)
            except Exception as e:
                print(f"    ERROR: {str(e)[:150]}", flush=True)
                results.append({"model": model, "dtype": dtype, "ok": False, "error": str(e)[:200]})
            finally:
                await page.close()

    (HERE / f"results_cdp_{args.device}.json").write_text(json.dumps(results, indent=2))
    print(f"\n| model | dtype | device | dim | load s | ms/post |")
    print(f"|---|---|---|---|---|---|")
    for r in results:
        if r.get("ok"):
            print(f"| {(r.get('modelId') or r.get('model','?')).split('/')[-1]} | {r['dtype']} | {r.get('actualDevice')} | "
                  f"{r['dim']} | {r['loadMs']/1000:.1f} | {r['warmMsPerPost']:.1f} |")
        else:
            print(f"| {(r.get('modelId') or r.get('model','?')).split('/')[-1]} | {r.get('dtype')} | - | - | - | FAILED |")
    httpd.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
