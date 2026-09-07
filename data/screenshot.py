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

        left = args.wait
        while left > 0:
            n = await page.evaluate("document.querySelectorAll('.CpftDupChip').length")
            if n:
                break
            await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
            await page.wait_for_timeout(2500)
            left -= 2.5

        chips = await page.evaluate("document.querySelectorAll('.CpftDupChip').length")
        if not chips:
            stats = await page.evaluate(
                "document.documentElement.getAttribute('data-cpftdup-posts')")
            print(f"no duplicate group appeared after scrolling {stats} posts.")
            print("Not a bug -- at tau=0.89 only ~2% of posts fold, so a quiet timeline")
            print("may simply not contain one. Re-run during a breaking news cycle, or")
            print("browse a trending topic page where the same story repeats.")
            await ctx.close()
            return 1

        # Bring the chip into view and frame it near the top, where a reviewer looks.
        await page.evaluate("""() => {
            const c = document.querySelector('.CpftDupChip')
            c.scrollIntoView({block: 'center'})
        }""")
        await page.wait_for_timeout(1200)
        await page.screenshot(path=str(OUT / "screenshot-1-collapsed.png"))
        print(f"wrote {OUT/'screenshot-1-collapsed.png'}")

        await page.evaluate("document.querySelector('.CpftDupChip').click()")
        await page.wait_for_timeout(900)
        await page.screenshot(path=str(OUT / "screenshot-2-expanded.png"))
        print(f"wrote {OUT/'screenshot-2-expanded.png'}")
        await ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
