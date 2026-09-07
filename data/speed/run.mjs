#!/usr/bin/env node

/**
 * In-browser latency benchmark driver for sentence-embedding models under transformers.js.
 * Plain ESM, Node 22, dependency-free.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const DIR = import.meta.dirname;
const RESULTS_FILE = path.resolve(DIR, 'results.json');
const POSTS_FILE = path.resolve(DIR, '../posts.jsonl');
const TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes per call

const ALL_MODELS = [
  'onnx-community/harrier-oss-v1-270m-ONNX',
  'onnx-community/gte-multilingual-base',
  'onnx-community/embeddinggemma-300m-ONNX',
  'jinaai/jina-embeddings-v5-text-nano-clustering',
  'onnx-community/F2LLM-v2-160M-ONNX',
  'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
  'hotchpotch/bekko-embedding-v1-a8m',
  'ibm-granite/granite-embedding-97m-multilingual-r2'
];

const ALL_DEVICES = ['webgpu', 'wasm'];

// 20 built-in fallback sample strings (mix English and Chinese, all length >= 40 chars)
const BUILTIN_FALLBACK_TEXTS = [
  "The open-source AI community continues to push the boundaries of on-device inference with smaller, faster embedding models.",
  "Breaking news: Researchers announce a major breakthrough in quantum computing error correction algorithms today.",
  "Transformers.js brings state-of-the-art machine learning models directly into web browsers without any server backend.",
  "WebGPU enables hardware-accelerated compute shaders directly on the client, drastically improving web application latency.",
  "Today's timeline deduplication benchmark compares embedding quality, memory consumption, and inference speed across architectures.",
  "I just released an updated version of the browser extension with custom filtering and timeline organization options.",
  "Modern neural network architectures for text embeddings balance semantic representation fidelity against parameter footprint.",
  "Performance profiling revealed that memory bandwidth and kernel launch overhead dominate latency for tiny transformer layers.",
  "Decentralized social networks are gaining steady traction as users seek greater privacy and algorithmic sovereignty.",
  "Exploring vector embeddings for cross-lingual semantic search and document retrieval across multilingual corpora.",
  "开源人工智能社区不断推动端侧推理的边界，涌现出越来越轻量且高效的句向量嵌入模型。",
  "重大突破：研究团队今日宣布在量子计算量子纠错算法方面取得关键进展，为实用化奠定坚实基础。",
  "Transformers.js 让最前沿的机器学习模型能够直接在网页端运行，无需任何后端服务器支撑与通信开销。",
  "WebGPU 技术让客户端能够直接利用现代显卡的强大算力进行通用计算，显著降低了端侧推理延迟与能耗。",
  "今日的时间线去重基准测试旨在全面对比各种模型架构在嵌入质量、内存占用以及端侧推理速度上的表现。",
  "我刚刚发布了新版浏览器扩展，新增了自定义过滤规则以及多语言推文语义聚类整理的强大功能。",
  "现代文本嵌入模型的神经网络结构在语义表征保真度与模型参数规模之间实现了精妙而高效的平衡。",
  "深入的性能剖析表明，对于小型 Transformer 层而言，显存带宽与内核启动开销是推理延迟的主要来源。",
  "去中心化社交网络正获得越来越多的关注，越来越多的用户开始重视个人隐私保护和内容推荐的自主掌控权。",
  "探索跨语言语义检索与多语言语料库文档对齐的高维向量嵌入技术，以及在端侧设备上的实际工程应用落地。"
];

/**
 * Parse CLI arguments: --models <list> and --device <dev>
 */
function parseCliArgs() {
  const args = process.argv.slice(2);
  let modelsArg = null;
  let deviceArg = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      console.log(`Usage: node run.mjs [options]

Options:
  --models <list>   Comma-separated model names or substrings (default: all 8 models)
  --device <dev>    Device to run: webgpu or wasm (default: both)
  --help, -h        Show this help message
`);
      process.exit(0);
    }
    if (arg === '--models') {
      modelsArg = args[++i];
    } else if (arg.startsWith('--models=')) {
      modelsArg = arg.slice('--models='.length);
    } else if (arg === '--device') {
      deviceArg = args[++i];
    } else if (arg.startsWith('--device=')) {
      deviceArg = arg.slice('--device='.length);
    }
  }

  let selectedModels = ALL_MODELS;
  if (modelsArg) {
    const queries = modelsArg.split(',').map(s => s.trim()).filter(Boolean);
    const matched = [];
    for (const q of queries) {
      const hits = ALL_MODELS.filter(m => m === q || m.toLowerCase().includes(q.toLowerCase()));
      if (hits.length > 0) {
        for (const h of hits) {
          if (!matched.includes(h)) matched.push(h);
        }
      } else {
        matched.push(q);
      }
    }
    selectedModels = matched;
  }

  let selectedDevices = ALL_DEVICES;
  if (deviceArg) {
    const d = deviceArg.trim().toLowerCase();
    if (!ALL_DEVICES.includes(d)) {
      console.warn(`[warning] Unknown device '${d}', proceeding with '${d}' anyway.`);
    }
    selectedDevices = [d];
  }

  return { selectedModels, selectedDevices };
}

/**
 * Read benchmark texts from ../posts.jsonl or fall back to built-in sample strings.
 * Takes the first 20 with length >= 40 chars.
 */
function loadBenchmarkTexts() {
  const texts = [];
  if (fs.existsSync(POSTS_FILE)) {
    try {
      const content = fs.readFileSync(POSTS_FILE, 'utf8');
      const lines = content.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const item = JSON.parse(trimmed);
          if (typeof item.text === 'string') {
            const t = item.text.trim();
            if (t.length >= 40) {
              texts.push(t);
              if (texts.length === 20) break;
            }
          }
        } catch {}
      }
      console.log(`[texts] Loaded ${texts.length} post(s) (>= 40 chars) from ${POSTS_FILE}`);
    } catch (err) {
      console.warn(`[warning] Could not read ${POSTS_FILE}: ${err.message}`);
    }
  } else {
    console.log(`[texts] ${POSTS_FILE} not found.`);
  }

  if (texts.length < 20) {
    console.log(`[texts] Using built-in sample texts to reach 20 samples (needed ${20 - texts.length} more).`);
    let idx = 0;
    while (texts.length < 20 && idx < BUILTIN_FALLBACK_TEXTS.length) {
      texts.push(BUILTIN_FALLBACK_TEXTS[idx++]);
    }
  }

  return texts.slice(0, 20);
}

/**
 * Start a minimal HTTP server for data/speed/ on an ephemeral port.
 */
function startStaticServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      try {
        const reqUrl = new URL(req.url, 'http://127.0.0.1');
        let pathname = decodeURIComponent(reqUrl.pathname);
        if (pathname === '/') pathname = '/index.html';

        const filePath = path.normalize(path.join(DIR, pathname));
        if (!filePath.startsWith(DIR)) {
          res.writeHead(403);
          return res.end('Forbidden');
        }

        if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
          res.writeHead(404);
          return res.end('Not Found');
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentTypes = {
          '.html': 'text/html; charset=utf-8',
          '.js': 'application/javascript; charset=utf-8',
          '.mjs': 'application/javascript; charset=utf-8',
          '.json': 'application/json; charset=utf-8',
          '.css': 'text/css; charset=utf-8',
          '.txt': 'text/plain; charset=utf-8',
        };

        const contentType = contentTypes[ext] || 'application/octet-stream';
        res.writeHead(200, {
          'Content-Type': contentType,
          'Access-Control-Allow-Origin': '*',
          'Cache-Control': 'no-cache, no-store, must-revalidate',
        });
        fs.createReadStream(filePath).pipe(res);
      } catch (err) {
        res.writeHead(500);
        res.end(String(err));
      }
    });

    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      resolve({ server, port, url: `http://127.0.0.1:${port}/index.html` });
    });
    server.on('error', reject);
  });
}

/**
 * Locate Chrome executable on the system.
 */
function findChromeExecutable() {
  if (process.env.CHROME_PATH && fs.existsSync(process.env.CHROME_PATH)) {
    return process.env.CHROME_PATH;
  }
  const candidates = [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return undefined;
}

/**
 * Probe playwright-core / puppeteer-core or fall back to CDP.
 */
async function launchOrConnectBrowser(benchUrl) {
  const chromeArgs = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-sandbox',
    '--disable-dev-shm-usage',
  ];

  // 1. Try playwright-core
  let pw = null;
  for (const specifier of [
    'playwright-core',
    'playwright',
    path.resolve(DIR, '../../.venv/lib/python3.12/site-packages/playwright/driver/package/index.mjs'),
    path.resolve(DIR, '../../.venv/lib/python3.11/site-packages/playwright/driver/package/index.mjs'),
  ]) {
    try {
      pw = await import(specifier);
      if (pw && (pw.chromium || pw.default?.chromium)) break;
    } catch {}
  }

  const chromium = pw?.chromium || pw?.default?.chromium;
  if (chromium) {
    console.log('[browser] Found playwright-core. Launching headless Chrome with WebGPU flags...');
    const execPath = findChromeExecutable();
    const launchOptions = {
      headless: true,
      args: chromeArgs
    };
    if (execPath) {
      launchOptions.executablePath = execPath;
    } else {
      launchOptions.channel = 'chrome';
    }

    const browser = await chromium.launch(launchOptions);
    const page = await browser.newPage();

    const waitForReady = async (maxWaitMs = 45000) => {
      await page.waitForFunction(() => window.BENCH_READY === true, { timeout: maxWaitMs });
    };

    return {
      type: 'playwright-core',
      navigate: async () => {
        await page.goto(benchUrl, { waitUntil: 'load', timeout: 45000 });
        await waitForReady();
      },
      reload: async () => {
        await page.reload({ waitUntil: 'load', timeout: 45000 });
        await waitForReady();
      },
      runBench: async (modelId, dtype, device, texts, timeoutMs) => {
        const runPromise = page.evaluate(
          ({ modelId, dtype, device, texts }) => window.runBench(modelId, dtype, device, texts),
          { modelId, dtype, device, texts }
        );

        const timeoutPromise = new Promise(resolve => {
          setTimeout(() => {
            resolve({
              modelId, dtype, device, actualDevice: null,
              loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
              ok: false,
              error: `Benchmark call timed out after ${timeoutMs / 1000}s`
            });
          }, timeoutMs);
        });

        try {
          return await Promise.race([runPromise, timeoutPromise]);
        } catch (err) {
          return {
            modelId, dtype, device, actualDevice: null,
            loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
            ok: false,
            error: String(err)
          };
        }
      },
      close: async () => {
        try { await browser.close(); } catch {}
      }
    };
  }

  // 2. Try puppeteer-core
  let puppeteer = null;
  for (const specifier of ['puppeteer-core', 'puppeteer']) {
    try {
      puppeteer = await import(specifier);
      if (puppeteer && (puppeteer.launch || puppeteer.default?.launch)) break;
    } catch {}
  }

  const pLaunch = puppeteer?.launch || puppeteer?.default?.launch;
  if (pLaunch) {
    console.log('[browser] Found puppeteer-core. Launching headless Chrome with WebGPU flags...');
    const execPath = findChromeExecutable();
    const launchOptions = {
      headless: true,
      args: chromeArgs
    };
    if (execPath) {
      launchOptions.executablePath = execPath;
    } else {
      launchOptions.channel = 'chrome';
    }

    const browser = await pLaunch.call(puppeteer, launchOptions);
    const page = await browser.newPage();

    const waitForReady = async (maxWaitMs = 45000) => {
      await page.waitForFunction(() => window.BENCH_READY === true, { timeout: maxWaitMs });
    };

    return {
      type: 'puppeteer-core',
      navigate: async () => {
        await page.goto(benchUrl, { waitUntil: 'load', timeout: 45000 });
        await waitForReady();
      },
      reload: async () => {
        await page.reload({ waitUntil: 'load', timeout: 45000 });
        await waitForReady();
      },
      runBench: async (modelId, dtype, device, texts, timeoutMs) => {
        const runPromise = page.evaluate(
          (modelId, dtype, device, texts) => window.runBench(modelId, dtype, device, texts),
          modelId, dtype, device, texts
        );

        const timeoutPromise = new Promise(resolve => {
          setTimeout(() => {
            resolve({
              modelId, dtype, device, actualDevice: null,
              loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
              ok: false,
              error: `Benchmark call timed out after ${timeoutMs / 1000}s`
            });
          }, timeoutMs);
        });

        try {
          return await Promise.race([runPromise, timeoutPromise]);
        } catch (err) {
          return {
            modelId, dtype, device, actualDevice: null,
            loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
            ok: false,
            error: String(err)
          };
        }
      },
      close: async () => {
        try { await browser.close(); } catch {}
      }
    };
  }

  // 3. Fall back to CDP against an already-running Chrome on 127.0.0.1:9222
  console.log('[browser] Neither playwright-core nor puppeteer-core found.');
  console.log('[browser] Falling back to CDP against Chrome on 127.0.0.1:9222...');

  let versionInfo;
  try {
    const res = await fetch('http://127.0.0.1:9222/json/version');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    versionInfo = await res.json();
  } catch (err) {
    throw new Error(
      `Could not connect to Chrome on 127.0.0.1:9222 (${err.message}).\n` +
      `Please ensure Chrome is running with:\n` +
      `  google-chrome --remote-debugging-port=9222 --enable-unsafe-webgpu --enable-features=Vulkan\n` +
      `or install playwright-core.`
    );
  }

  console.log(`[cdp] Connected to ${versionInfo.Browser || 'Chrome'}`);

  // Create a new tab for the benchmark
  const tab = await fetch('http://127.0.0.1:9222/json/new?' + encodeURIComponent(benchUrl), {
    method: 'PUT'
  }).then(r => r.json());

  const tabId = tab.id;
  const ws = new WebSocket(tab.webSocketDebuggerUrl);

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  let nextId = 1;
  const pendingRequests = new Map();
  let loadEventResolver = null;

  ws.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.method === 'Page.loadEventFired' && loadEventResolver) {
        const resolve = loadEventResolver;
        loadEventResolver = null;
        resolve();
      }
      if (data.id && pendingRequests.has(data.id)) {
        const { resolve, reject } = pendingRequests.get(data.id);
        pendingRequests.delete(data.id);
        if (data.error) {
          reject(new Error(data.error.message || JSON.stringify(data.error)));
        } else {
          resolve(data.result);
        }
      }
    } catch {}
  });

  const sendCdp = (method, params = {}) => {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pendingRequests.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  };

  await sendCdp('Page.enable');
  await sendCdp('Runtime.enable');

  const waitForBenchReady = async (maxWaitMs = 45000) => {
    const start = Date.now();
    while (Date.now() - start < maxWaitMs) {
      try {
        const evalRes = await sendCdp('Runtime.evaluate', {
          expression: 'Boolean(window.BENCH_READY)',
          returnByValue: true
        });
        if (evalRes?.result?.value === true) return;
      } catch {}
      await new Promise(r => setTimeout(r, 150));
    }
    throw new Error('Timed out waiting for window.BENCH_READY in benchmark page');
  };

  await waitForBenchReady();

  return {
    type: 'cdp (127.0.0.1:9222)',
    navigate: async () => {
      const loadPromise = new Promise(r => { loadEventResolver = r; });
      await sendCdp('Page.navigate', { url: benchUrl });
      await loadPromise;
      await waitForBenchReady();
    },
    reload: async () => {
      const loadPromise = new Promise(r => { loadEventResolver = r; });
      await sendCdp('Page.reload', { ignoreCache: false });
      await loadPromise;
      await waitForBenchReady();
    },
    runBench: async (modelId, dtype, device, texts, timeoutMs) => {
      const expression = `window.runBench(${JSON.stringify(modelId)}, ${JSON.stringify(dtype)}, ${JSON.stringify(device)}, ${JSON.stringify(texts)})`;

      const evalPromise = sendCdp('Runtime.evaluate', {
        expression,
        awaitPromise: true,
        returnByValue: true,
        timeout: timeoutMs
      }).then(res => {
        if (res?.result?.value) {
          return res.result.value;
        }
        if (res?.exceptionDetails) {
          return {
            modelId, dtype, device, actualDevice: null,
            loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
            ok: false,
            error: res.exceptionDetails.text || JSON.stringify(res.exceptionDetails)
          };
        }
        return {
          modelId, dtype, device, actualDevice: null,
          loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
          ok: false,
          error: 'No result returned from runBench'
        };
      }).catch(err => ({
        modelId, dtype, device, actualDevice: null,
        loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
        ok: false,
        error: String(err)
      }));

      const timeoutPromise = new Promise(resolve => {
        setTimeout(() => {
          resolve({
            modelId, dtype, device, actualDevice: null,
            loadMs: null, warmMsPerPost: null, batchMs: null, dim: null,
            ok: false,
            error: `Benchmark call timed out after ${timeoutMs / 1000}s`
          });
        }, timeoutMs);
      });

      return await Promise.race([evalPromise, timeoutPromise]);
    },
    close: async () => {
      try { ws.close(); } catch {}
      try {
        await fetch(`http://127.0.0.1:9222/json/close/${tabId}`);
      } catch {}
    }
  };
}

/**
 * Print results as a GitHub-flavored markdown table.
 */
function printMarkdownTable(results) {
  const headers = ['model', 'dtype', 'device', 'dim', 'load s', 'ms/post', 'ok'];
  const rows = results.map(r => [
    r.modelId || r.model || '-',
    r.dtype || '-',
    r.device || '-',
    r.dim !== null && r.dim !== undefined ? String(r.dim) : '-',
    typeof r.loadMs === 'number' ? (r.loadMs / 1000).toFixed(2) : '-',
    typeof r.warmMsPerPost === 'number' ? r.warmMsPerPost.toFixed(2) : '-',
    String(Boolean(r.ok))
  ]);

  const colWidths = headers.map((h, i) => Math.max(h.length, ...rows.map(r => r[i].length)));
  const pad = (str, len) => str + ' '.repeat(Math.max(0, len - str.length));

  const headerLine = '| ' + headers.map((h, i) => pad(h, colWidths[i])).join(' | ') + ' |';
  const dividerLine = '| ' + colWidths.map(w => '-'.repeat(Math.max(3, w))).join(' | ') + ' |';
  const rowLines = rows.map(r => '| ' + r.map((cell, i) => pad(cell, colWidths[i])).join(' | ') + ' |');

  console.log('\n' + [headerLine, dividerLine, ...rowLines].join('\n') + '\n');
}

/**
 * Main benchmark execution loop.
 */
async function main() {
  const { selectedModels, selectedDevices } = parseCliArgs();
  const texts = loadBenchmarkTexts();

  console.log(`[setup] Selected ${selectedModels.length} model(s) and ${selectedDevices.length} device(s)`);
  console.log(`[setup] Benchmark texts count: ${texts.length}`);

  console.log('[setup] Starting static file server...');
  const { server, port, url: benchUrl } = await startStaticServer();
  console.log(`[server] Serving benchmark harness at ${benchUrl}`);

  let browserSession;
  try {
    browserSession = await launchOrConnectBrowser(benchUrl);
    console.log(`[browser] Ready using ${browserSession.type}`);
  } catch (err) {
    server.close();
    console.error(`[fatal] Failed to initialize browser: ${err.message}`);
    process.exit(1);
  }

  // Graceful cleanup handler
  let cleanedUp = false;
  const cleanup = async () => {
    if (cleanedUp) return;
    cleanedUp = true;
    console.log('\n[teardown] Cleaning up...');
    try { await browserSession?.close(); } catch {}
    try {
      if (server) {
        if (typeof server.closeAllConnections === 'function') {
          server.closeAllConnections();
        }
        server.close();
      }
    } catch {}
  };
  process.on('SIGINT', async () => { await cleanup(); process.exit(130); });
  process.on('SIGTERM', async () => { await cleanup(); process.exit(143); });

  const results = [];

  try {
    // Initial page load
    await browserSession.navigate();

    for (const device of selectedDevices) {
      console.log(`\n============================================================`);
      console.log(`BENCHMARK RUN: DEVICE = ${device.toUpperCase()}`);
      console.log(`============================================================`);

      for (const model of selectedModels) {
        console.log(`\n>>> Testing: ${model} on ${device}`);

        // Reload the page between models so no model stays resident in memory
        await browserSession.reload();

        // 1. Try dtype 'q8' FIRST
        console.log(`[attempt] Trying dtype 'q8'...`);
        let res = await browserSession.runBench(model, 'q8', device, texts, TIMEOUT_MS);

        // 2. If that fails, retry with 'fp32'
        if (!res.ok) {
          console.log(`[fallback] 'q8' failed (${res.error}).`);
          console.log(`[attempt] Retrying with dtype 'fp32'...`);

          // Reload page to ensure clean memory before retry
          await browserSession.reload();
          res = await browserSession.runBench(model, 'fp32', device, texts, TIMEOUT_MS);

          if (res.ok) {
            console.log(`[fallback] 'fp32' succeeded! (dim=${res.dim}, warmMsPerPost=${res.warmMsPerPost?.toFixed(2)})`);
          } else {
            console.log(`[fallback] 'fp32' also failed (${res.error}).`);
          }
        } else {
          console.log(`[success] 'q8' succeeded! (dim=${res.dim}, warmMsPerPost=${res.warmMsPerPost?.toFixed(2)})`);
        }

        const record = {
          model,
          modelId: model,
          dtype: res.dtype,
          device,
          actualDevice: res.actualDevice || device,
          dim: res.dim,
          loadMs: res.loadMs,
          loadSec: typeof res.loadMs === 'number' ? Number((res.loadMs / 1000).toFixed(2)) : null,
          batchMs: res.batchMs,
          warmMsPerPost: typeof res.warmMsPerPost === 'number' ? Number(res.warmMsPerPost.toFixed(2)) : null,
          ok: res.ok,
          error: res.error
        };

        results.push(record);

        // Progressively write results.json after every model
        fs.writeFileSync(RESULTS_FILE, JSON.stringify(results, null, 2) + '\n', 'utf8');
      }
    }

    console.log(`\n============================================================`);
    console.log(`BENCHMARK COMPLETE`);
    console.log(`============================================================`);
    console.log(`Wrote ${results.length} result(s) to ${RESULTS_FILE}`);

    // Print final markdown table to stdout
    printMarkdownTable(results);

  } finally {
    await cleanup();
    process.exit(0);
  }
}

main().catch(err => {
  console.error('[fatal]', err);
  process.exit(1);
});
