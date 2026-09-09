#!/usr/bin/env python3
"""
Capture Chrome Web Store screenshots of the extension actually working.

Screenshots are the only submission asset that cannot be produced from the repository,
because they have to show real folded posts on a real timeline. This loads the store
build against a logged-in session, scrolls until a duplicate group appears, and captures
the group both collapsed and expanded -- reviewers respond much better to seeing that
nothing is destroyed.

Runs HEADED at exactly 1280x800, the store's preferred size. Headless is not an option:
Chrome pauses requestAnimationFrame in hidden tabs, and the observer schedules its DOM
writes inside rAF, so a headless run computes the right clusters and then silently drops
every write -- you would screenshot an empty timeline and not know why.
"""
import argparse, asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "dist-store"
OUT = REPO / "store" / "store-assets"
PROFILE = Path.home() / ".cache" / "x-dedup-shot"
COOKIES = Path("/tmp/x_cookies.json")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=300, help="seconds to hunt for a duplicate")
    ap.add_argument("--warm", action="store_true",
                    help="just browse and remember, take no shots -- run this first so the "
                         "next run has a memory to collapse against")
    ap.add_argument("--keep-profile", action="store_true", default=True)
    ap.add_argument("--url", default="https://x.com/home")
    args = ap.parse_args()

    if not EXT.exists():
        sys.exit(f"missing {EXT} -- run ./scripts/build-store.sh first")
    if not COOKIES.exists():
        sys.exit(f"missing {COOKIES} -- export cookies from a logged-in Chrome first")
    OUT.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}",
                  "--enable-features=Vulkan", "--enable-unsafe-webgpu", "--no-first-run"],
            viewport={"width": 1280, "height": 800},
        )
        await ctx.add_cookies(json.loads(COOKIES.read_text()))
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)

        # Browse, then RELOAD in the same session and shoot the collapsed state.
        #
        # Two passes in one browser, not two runs: closing the context can beat the
        # debounced flush to disk, so a fresh run may start with an empty memory and
        # nothing to collapse. Reloading keeps the service worker (and its IndexedDB)
        # alive across the boundary, which is also exactly what a reader does.
        for _ in range(8):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await page.wait_for_timeout(1600)
        await page.wait_for_timeout(13000)          # past the 10s flush debounce
        seen1 = await page.evaluate(
            "document.documentElement.getAttribute('data-cpftdup-posts')")
        print(f"pass 1: {seen1} posts remembered")

        await page.reload(wait_until="domcontentloaded")
        await page.wait_for_timeout(12000)

        # Do NOT scroll hunting for a control after the reload. The remembered posts are
        # the ones at the top, already collapsed by the time the page settles, and
        # scrolling past them lets X recycle their cells -- the control is found and then
        # gone before the shutter. Settle, then shoot.
        await page.wait_for_timeout(4000)
        chips = await page.evaluate("document.querySelectorAll('.CpftDupChip').length")
        if not chips:
            stats = await page.evaluate(
                "document.documentElement.getAttribute('data-cpftdup-remembered')")
            print(f"nothing collapsed after the reload ({stats} posts remembered).")
            print("The page served different posts the second time; re-run, or use a URL")
            print("whose results are stable between loads.")
            await ctx.close()
            return 1

        # Re-query rather than trusting the poll: X re-renders constantly, so a chip that
        # existed a moment ago may already be gone. Retry a few times before giving up.
        placed = False
        for _ in range(10):
            placed = await page.evaluate("""() => {
                const c = document.querySelector('.CpftDupChip')
                if (!c) return false
                c.scrollIntoView({block: 'center'})
                return true
            }""")
            if placed:
                break
            await page.wait_for_timeout(1500)
        if not placed:
            print("a control appeared during scrolling but was gone by capture time; re-run")
            await ctx.close()
            return 1
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(OUT / "screenshot-1-collapsed.png"))
        print(f"wrote {OUT/'screenshot-1-collapsed.png'}")

        await page.evaluate("document.querySelector('.CpftDupChip')?.click()")
        await page.wait_for_timeout(1200)
        await page.screenshot(path=str(OUT / "screenshot-2-expanded.png"))
        print(f"wrote {OUT/'screenshot-2-expanded.png'}")
        await ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
