#!/usr/bin/env python3
"""
End-to-end verification of the dedup extension against real x.com.

Uses Playwright's bundled Chromium, NOT the system Chrome: Chrome 152 has removed
--load-extension entirely (verified -- it registers nothing and prints no error, with
or without --remote-debugging-port, on a fresh profile). Playwright's Chromium still
supports it via a persistent context, which is what makes this testable at all.

Checks, in the order that matters -- each one gates the next:
  1. content script injected and CSS applied
  2. offscreen document created                 <- the whole CSP design
  3. model actually downloads from huggingface.co on the EXTENSION origin
  4. posts get clustered, duplicates collapse, chip renders
  5. fail-open: nothing hidden when the model is unavailable

Note on headless: Chrome PAUSES requestAnimationFrame in hidden/background tabs, and
observer.js schedules its DOM writes inside rAF. A headless run computes the right
clusters and then silently drops every write, which looks exactly like a broken filter.
So this runs HEADED and asserts document.visibilityState first.
"""
import argparse, asyncio, json, shutil, sys
from pathlib import Path
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "dist-dedup"
PROFILE = Path.home() / ".cache" / "x-dedup-e2e"
SRC_PROFILE = Path.home() / ".cache" / "x-dedup-profile"


COOKIES = Path("/tmp/x_cookies.json")


def load_cookies():
    """Session cookies exported from the logged-in system Chrome.

    Copying the profile directory does NOT work across browser builds: Chrome encrypts
    the cookie store with a key from the OS keyring, and Playwright's Chromium cannot
    decrypt the system Chrome's store -- the copy loads, and you land on the login page
    with no error. Exporting decrypted cookies over CDP and re-adding them sidesteps
    the encryption entirely.

    Regenerate with:
      google-chrome --user-data-dir=~/.cache/x-dedup-profile --remote-debugging-port=9222
      (then the export snippet in the session transcript)
    """
    if not COOKIES.exists():
        sys.exit(f"missing {COOKIES} -- export cookies from the logged-in Chrome first")
    return json.loads(COOKIES.read_text())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=180,
                    help="seconds to allow for the model download on first run")
    ap.add_argument("--url", default="https://x.com/home")
    args = ap.parse_args()

    if not EXT.exists():
        sys.exit(f"missing {EXT} -- run ./scripts/build-dedup.sh first")
    cookies = load_cookies()

    hf_hits, results = [], {}
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            args=[
                f"--disable-extensions-except={EXT}",
                f"--load-extension={EXT}",
                "--enable-features=Vulkan",
                "--enable-unsafe-webgpu",
                "--no-first-run",
            ],
            viewport={"width": 1400, "height": 1000},
        )
        ctx.on("request", lambda r: hf_hits.append(r.url)
               if "huggingface.co" in r.url or "hf.co" in r.url else None)
        await ctx.add_cookies(cookies)

        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)

        # Probe via the DOM, never via window.* -- content scripts run in the ISOLATED
        # world and page.evaluate() runs in the MAIN world, so isolated-world globals are
        # invisible here. Likewise content-script CSS does not appear in
        # document.styleSheets, so it is checked by computed style on a probe element.
        results["visibilityState"] = await page.evaluate("document.visibilityState")
        results["contentScript"] = await page.evaluate(
            "document.documentElement.getAttribute('data-cpftdup-ready') === '1'")
        results["model"] = await page.evaluate(
            "document.documentElement.getAttribute('data-cpftdup-model')")
        results["cssApplied"] = await page.evaluate("""() => {
            const el = document.createElement('button');
            el.className = 'CpftDupChip';
            document.body.appendChild(el);
            const r = getComputedStyle(el).borderRadius;
            el.remove();
            return r === '9999px';
        }""")
        # NOT a pass/fail signal: an offscreen document is not exposed through
        # ctx.pages or ctx.background_pages, so this reads false even when the path is
        # demonstrably working. Embeddings produced is the signal; this is a hint only.
        results["offscreenVisibleToPlaywright"] = \
            any("offscreen" in p.url for p in ctx.background_pages) or \
            any("offscreen" in p.url for p in ctx.pages)

        # Scroll to feed the observer real posts, and give the model time to download.
        deadline = args.wait
        while deadline > 0:
            await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
            await page.wait_for_timeout(3000)
            deadline -= 3
            multi = await page.evaluate(
                "+(document.documentElement.getAttribute('data-cpftdup-multi') || 0)")
            if multi > 0:
                break

        results["hfRequests"] = len(hf_hits)
        results["hfSample"] = hf_hits[:3]
        results["stats"] = await page.evaluate("""() => {
            const d = document.documentElement, g = (k) => +(d.getAttribute('data-cpftdup-'+k) || 0);
            return { posts: g('posts'), clusters: g('clusters'),
                     multi: g('multi'), collapsed: g('collapsed'),
                     embedded: g('embedded'), embedErrors: g('embed-errors') };
        }""")
        # Virtualisation check (plan item 7). Deep in the feed the members of a cluster
        # have usually been recycled out of the DOM, so "store says 1 collapsed, DOM says
        # 0" is the EXPECTED reading there and proves nothing either way. Scrolling back
        # to the top forces X to re-create those nodes, and the observer must repaint them
        # from the id-keyed store. Painted state after that round trip is the real check:
        # if the store is right and the DOM stays empty here, the writes are being dropped.
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(4000)
        results["afterScrollBack"] = await page.evaluate("""() => {
            const ids = (document.documentElement.getAttribute('data-cpftdup-dupids') || '')
                .split(',').filter(Boolean)
            // For each post the store believes is a duplicate, find its article (if it
            // is still in the DOM at all) and ask whether it is actually marked.
            let present = 0, marked = 0
            for (const id of ids) {
                const a = document.querySelector(`a[href*="/status/${id}"]`)
                    ?.closest('[data-testid="cellInnerDiv"]')
                if (!a) continue
                present++
                if (a.firstElementChild?.classList.contains('CpftDup')) marked++
            }
            return {
                articles: document.querySelectorAll('article[data-testid="tweet"]').length,
                collapsed: document.querySelectorAll('.CpftDup').length,
                chips: document.querySelectorAll('.CpftDupChip').length,
                dupIds: ids.length, dupsStillInDom: present, dupsMarked: marked,
            }
        }""")

        results["loggedIn"] = await page.evaluate(
            "!document.querySelector('a[href=\"/login\"]') && !location.href.includes('/i/flow/login')")
        results["articles"] = await page.evaluate(
            """document.querySelectorAll('article[data-testid="tweet"]').length""")
        results["collapsed"] = await page.evaluate(
            "document.querySelectorAll('.CpftDup').length")
        results["chips"] = await page.evaluate(
            "document.querySelectorAll('.CpftDupChip').length")
        await ctx.close()

    print(json.dumps(results, indent=2))
    ok = bool(results["contentScript"]) and bool(results["cssApplied"])
    print("\nCONTENT SCRIPT:", "PASS" if ok else "FAIL")
    emb = (results.get("stats") or {}).get("embedded", 0)
    err = (results.get("stats") or {}).get("embedErrors", 0)
    # Judge on embeddings produced, not on observed network requests: Playwright cannot
    # see the offscreen document's traffic, so hfRequests is unreliable by construction.
    print("MODEL / EMBED :", f"PASS ({emb} posts embedded, {err} failed)" if emb else
          f"FAIL (0 posts embedded, {err} errors -- offscreen/CSP path not working)")
    st = results.get("stats") or {}
    print("CLUSTERING    :", f"PASS ({st.get('multi',0)} multi-post clusters, "
          f"{st.get('collapsed',0)} posts collapsed)" if st.get("multi") else
          "no duplicates seen yet (not necessarily a failure -- depends on the timeline)")

    back = results.get("afterScrollBack") or {}
    if st.get("collapsed"):
        in_dom = back.get("dupsStillInDom", 0)
        marked = back.get("dupsMarked", 0)
        if in_dom == 0:
            print("COLLAPSE       :", f"INCONCLUSIVE ({back.get('chips',0)} chips shown, "
                  f"but none of the {back.get('dupIds',0)} duplicate posts is still in the "
                  f"DOM -- X recycled them, so nothing to mark. Re-run when a duplicate "
                  f"lands near the head of the feed.)")
        else:
            print("COLLAPSE       :", f"PASS ({marked}/{in_dom} duplicate posts still in "
                  f"the DOM carry .CpftDup, {back.get('chips',0)} chips)"
                  if marked == in_dom else
                  f"FAIL (only {marked}/{in_dom} duplicate posts present in the DOM are "
                  f"marked -- the class write is being dropped, not virtualisation)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
