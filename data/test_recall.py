#!/usr/bin/env python3
"""
Prove cross-session memory works in a real browser, not against a fake transport.

Loads x.com, scrolls to embed some posts, waits for the debounced flush, then RELOADS and
checks that posts come back from IndexedDB. The unit tests cover the logic; this covers
the parts they cannot: that the content script can reach the service worker, that the
service worker's IndexedDB survives a page reload, and that Int8Array survives structured
cloning in both directions.
"""
import asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "dist-store"
PROFILE = Path.home() / ".cache" / "x-dedup-recall"
COOKIES = Path("/tmp/x_cookies.json")


async def main():
    if not COOKIES.exists():
        sys.exit("missing /tmp/x_cookies.json")
    # Fresh profile each run: a leftover IndexedDB from a previous run would make the
    # recall assertion pass without this session having stored anything.
    import shutil
    shutil.rmtree(PROFILE, ignore_errors=True)

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}",
                  "--enable-features=Vulkan", "--enable-unsafe-webgpu", "--no-first-run"],
            viewport={"width": 1400, "height": 1000},
        )
        await ctx.add_cookies(json.loads(COOKIES.read_text()))
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        async def stats():
            return await page.evaluate("""() => {
                const d = document.documentElement, g = (k) => +(d.getAttribute('data-cpftdup-'+k) || 0)
                return { posts: g('posts'), embedded: g('embedded'), remembered: g('remembered') }
            }""")

        await page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(9000)
        for _ in range(12):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await page.wait_for_timeout(2000)
        first = await stats()
        print(f"session 1: {first['posts']} posts, {first['embedded']} embedded, "
              f"{first['remembered']} remembered")

        # The flush is debounced at 10s; visibilitychange also forces it. Wait past both.
        await page.wait_for_timeout(14000)

        await page.reload(wait_until="domcontentloaded")
        await page.wait_for_timeout(12000)
        second = await stats()
        print(f"session 2 (after reload): {second['posts']} posts, "
              f"{second['embedded']} embedded, {second['remembered']} remembered")
        await ctx.close()

    ok = first["embedded"] > 0 and second["remembered"] > 0
    print("\nCROSS-SESSION RECALL:", f"PASS ({second['remembered']} posts recalled from "
          f"IndexedDB after reload)" if ok else
          f"FAIL (embedded {first['embedded']} before reload, recalled "
          f"{second['remembered']} after -- nothing survived)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
