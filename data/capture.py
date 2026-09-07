#!/usr/bin/env python3
"""
Capture a diverse corpus of real X/Twitter posts for the dedup benchmark.

Attaches to an ALREADY-RUNNING, already-logged-in Chrome over CDP -- it never
handles credentials and never types into the page. Start Chrome with:

    google-chrome --remote-debugging-port=9222

Diversity is the point. A naive scrape of one /home session during one news cycle
produces a corpus where three stories own half the posts, and a benchmark on that
measures "can the model find the Fed story" rather than "can it tell stories apart".
So we sample across Following / For You / Explore sections / individual trend pages,
and hard-cap the contribution of any single source or trend.

Output: data/posts.jsonl
"""
import argparse, asyncio, json, random, re, sys
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).parent / "posts.jsonl"
CDP = "http://127.0.0.1:9222"

# Extracts every tweet currently in the DOM. Runs after each scroll step because X
# virtualises the timeline -- nodes are recycled and destroyed, so anything not read
# before it scrolls out of view is gone.
EXTRACT_JS = r"""
() => {
  const out = [];
  for (const art of document.querySelectorAll('article[data-testid="tweet"]')) {
    try {
      // Permalink anchor carries both the status id and the timestamp.
      const timeEl = art.querySelector('a[href*="/status/"] time');
      const permalink = timeEl?.closest('a');
      const m = permalink?.getAttribute('href')?.match(/\/([^\/]+)\/status\/(\d+)/);
      if (!m) continue;
      const statusId = m[2];
      const authorId = m[1];

      const textEl = art.querySelector('div[data-testid="tweetText"]');
      const text = textEl ? textEl.innerText : '';

      // A quoted tweet is a nested role=link containing a *different* status id.
      let quotedId = null;
      for (const a of art.querySelectorAll('div[role="link"] a[href*="/status/"]')) {
        const q = a.getAttribute('href').match(/\/status\/(\d+)/);
        if (q && q[1] !== statusId) { quotedId = q[1]; break; }
      }

      // External links: card target + any t.co/http anchor that isn't an X internal link.
      const urls = new Set();
      const card = art.querySelector('a[data-testid="card.wrapper"], div[data-testid="card.wrapper"] a');
      if (card?.href) urls.add(card.href);
      for (const a of art.querySelectorAll('a[href^="http"]')) {
        const h = a.href;
        if (!/(^https?:\/\/(x|twitter)\.com)/.test(h)) urls.add(h);
      }

      // Media keys: the opaque id in pbs.twimg.com/media/<KEY>. Same image reposted
      // by N accounts yields the same key -- a free exact-match dedup signal.
      const mediaKeys = new Set();
      for (const img of art.querySelectorAll('img[src*="twimg.com/media/"]')) {
        const k = img.src.match(/\/media\/([A-Za-z0-9_\-]+)/);
        if (k) mediaKeys.add(k[1]);
      }

      out.push({
        statusId, authorId, text,
        ts: timeEl?.getAttribute('datetime') || null,
        quotedId,
        urls: [...urls],
        mediaKeys: [...mediaKeys],
        isRepost: !!art.querySelector('[data-testid="socialContext"]'),
        hasCard: !!card,
      });
    } catch (e) { /* one bad node must not kill the sweep */ }
  }
  return out;
}
"""

TREND_JS = r"""
() => {
  // A real trend cell reads like:
  //   ['2', '·', 'Sports · Trending', '#AEWDynamite', 'Trending with All Out, Ospreay']
  // i.e. rank, separator, category, NAME, and sometimes a related-terms line.
  // A promoted cell has no rank and a 'Promoted by ...' line -- those are ads, not trends.
  const names = [];
  for (const el of document.querySelectorAll('[data-testid="trend"]')) {
    const lines = el.innerText.split('\n').map(s => s.trim()).filter(Boolean);
    if (lines.some(l => /Promoted by/i.test(l))) continue;
    const name = lines.find(l =>
      !/^\d+$/.test(l) &&                                   // rank
      l !== '·' &&                                     // separator
      !/Trending|Promoted|LIVE/i.test(l) &&                 // category + "Trending with ..."
      !/^\d[\d.,]*\s*(K|M)?\s+(posts|Posts|tweets)/.test(l) &&
      l.length > 1 && l.length < 60
    );
    if (name) names.push(name);
  }
  return [...new Set(names)];
}
"""


async def harvest(page, store, source, trend, cap, scrolls, pause):
    """Scroll a page, polling the DOM after each step, until cap or scrolls exhausted."""
    got = 0
    seen_here = set()
    stale = 0
    for i in range(scrolls):
        try:
            posts = await page.evaluate(EXTRACT_JS)
        except Exception as e:
            print(f"    ! extract failed: {e}", file=sys.stderr)
            break
        new = 0
        for p in posts:
            sid = p["statusId"]
            if sid in seen_here:
                continue
            seen_here.add(sid)
            if sid in store:  # already captured from another source
                continue
            if len((p["text"] or "").strip()) < 15:
                continue
            p["source"] = source
            p["trend"] = trend
            store[sid] = p
            new += 1
            got += 1
            if got >= cap:
                print(f"    {source}/{trend or '-'}: +{got} (cap)")
                return got
        stale = stale + 1 if new == 0 else 0
        if stale >= 3:  # timeline stopped yielding anything new
            break
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.85)")
        await page.wait_for_timeout(pause)
    print(f"    {source}/{trend or '-'}: +{got}")
    return got


async def goto(page, url, settle=2500):
    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    await page.wait_for_timeout(settle)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=60, help="cap per home/explore section")
    ap.add_argument("--per-trend", type=int, default=25, help="cap per individual trend page")
    ap.add_argument("--trends", type=int, default=10, help="how many trend pages to visit")
    ap.add_argument("--scrolls", type=int, default=25)
    ap.add_argument("--pause", type=int, default=1300, help="ms between scroll steps")
    ap.add_argument("--nav-delay", type=int, default=0,
                    help="ms to wait between search navigations. X rate-limits search "
                         "aggressively -- ~30 rapid navigations tripped it during the first "
                         "sweep and every subsequent page returned zero posts. Pace to avoid it.")
    ap.add_argument("--queries", default="",
                    help="comma-separated search terms to harvest INSTEAD of the standard "
                         "sweep. Used to top up under-represented languages: X's /explore "
                         "trends are locale-pinned (US here), so a natural sweep yields "
                         "almost no Chinese, which would make the multilingual comparison "
                         "untestable.")
    args = ap.parse_args()

    store = {}
    if OUT.exists():  # resume / accumulate across runs for temporal diversity
        for line in OUT.read_text().split("\n"):
            if line.strip():
                p = json.loads(line)
                store[p["statusId"]] = p
        print(f"resuming with {len(store)} existing posts")

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.connect_over_cdp(CDP)
        except Exception as e:
            print(f"\nCannot reach Chrome at {CDP}: {e}\n\n"
                  f"Quit Chrome completely, then relaunch it with:\n"
                  f"  google-chrome --remote-debugging-port=9222\n", file=sys.stderr)
            return 1

        ctx = browser.contexts[0]
        page = await ctx.new_page()

        # --- 0. Explicit query mode (language top-up) -----------------------
        if args.queries:
            qs = [s.strip() for s in args.queries.split(",") if s.strip()]
            for qi, q in enumerate(qs):
                try:
                    if qi and args.nav_delay:
                        await page.wait_for_timeout(args.nav_delay)
                    await goto(page, f"https://x.com/search?q={q}&f=live")
                    await harvest(page, store, "query", q, args.per_trend,
                                  args.scrolls, args.pause)
                except Exception as e:
                    print(f"    ! query {q!r}: {e}", file=sys.stderr)
            await page.close()
            with OUT.open("w") as f:
                for p in store.values():
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"\nwrote {len(store)} posts -> {OUT}")
            return 0

        # --- 1. Home: For You, then Following -------------------------------
        await goto(page, "https://x.com/home")
        tabs = await page.query_selector_all('[role="tab"]')
        print(f"home: found {len(tabs)} timeline tabs")
        for idx, label in enumerate(["for_you", "following"]):
            if idx < len(tabs):
                try:
                    tabs = await page.query_selector_all('[role="tab"]')
                    await tabs[idx].click()
                    await page.wait_for_timeout(2500)
                except Exception as e:
                    print(f"    ! tab {label}: {e}", file=sys.stderr)
            await harvest(page, store, f"home:{label}", None,
                          args.per_source, args.scrolls, args.pause)

        # --- 2. Explore sections --------------------------------------------
        explore = [("trending", "https://x.com/explore/tabs/trending"),
                   ("news", "https://x.com/explore/tabs/news_unified"),
                   ("sports", "https://x.com/explore/tabs/sports_unified"),
                   ("entertainment", "https://x.com/explore/tabs/entertainment_unified")]
        trend_names = []
        for name, url in explore:
            try:
                await goto(page, url)
                if name == "trending":
                    # This tab is a pure trend list -- it renders no articles at all,
                    # so scrolling it for posts just burns 35s for nothing.
                    #
                    # Trend cells hydrate PROGRESSIVELY: measured 16 cells at 3s and 30
                    # at 5s on the same load. A fixed settle silently returned a partial
                    # list, and once returned 0, which turned the whole sweep into a
                    # no-op that still looked like a successful run. Poll until the count
                    # stops growing instead of guessing a delay.
                    prev, stable = -1, 0
                    for _ in range(20):
                        n = await page.evaluate(
                            'document.querySelectorAll(\'[data-testid="trend"]\').length')
                        if n and n == prev:
                            stable += 1
                            if stable >= 2:
                                break
                        else:
                            stable = 0
                        prev = n
                        await page.wait_for_timeout(1000)
                    trend_names = await page.evaluate(TREND_JS)
                    print(f"explore: {len(trend_names)} trends discovered: "
                          f"{', '.join(trend_names[:8])}")
                    continue
                await harvest(page, store, f"explore:{name}", None,
                              args.per_source, args.scrolls, args.pause)
            except Exception as e:
                print(f"    ! explore/{name}: {e}", file=sys.stderr)

        # --- 3. Individual trend pages (capped hard) ------------------------
        random.shuffle(trend_names)
        for ti, t in enumerate(trend_names[: args.trends]):
            try:
                if ti and args.nav_delay:
                    await page.wait_for_timeout(args.nav_delay)
                q = re.sub(r"\s+", " ", t).strip()
                await goto(page, f"https://x.com/search?q={q}&src=trend_click&f=live")
                await harvest(page, store, "trend", q, args.per_trend, 12, args.pause)
            except Exception as e:
                print(f"    ! trend {t!r}: {e}", file=sys.stderr)

        await page.close()

    with OUT.open("w") as f:
        for p in store.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(store)} posts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
