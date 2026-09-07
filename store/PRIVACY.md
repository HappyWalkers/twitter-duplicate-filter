# Privacy Policy — Timeline Dedup for X

_Last updated: 7 September 2026_

**Timeline Dedup does not collect, transmit, store, or sell any personal information.**

## What the extension does

It reads the text of posts already visible on your x.com timeline, converts each to a
numeric vector using a machine-learning model that runs entirely inside your browser, and
compares those vectors to find posts about the same story. Near-duplicates are collapsed
behind a "+N similar posts" control that you can expand at any time.

## What leaves your device

**Nothing.** The extension makes no network requests of any kind.

This is a verifiable claim, not a promise:

* The extension declares **no host permissions**, so Chrome will not allow it to contact
  any website even if it tried.
* The model is **bundled inside the extension package** rather than downloaded, so there
  is no model-fetch traffic on first run or afterwards.
* There is no analytics, telemetry, crash reporting, or remote configuration.

You can confirm this yourself: open DevTools → Network on x.com with the extension
enabled, and you will see no requests originating from it.

## What is stored, and where

Two things, both in Chrome's local extension storage on your own computer, and neither
ever transmitted:

* whether the extension is switched on;
* a small counter of how many posts have been seen and folded, shown in the popup.

Post text and the vectors computed from it are held **in memory only**, for at most the
most recent 400 posts, and are discarded when the tab closes. Nothing is written to disk.

## Permissions, and why each is needed

* **`offscreen`** — creates a hidden extension page that hosts the model. Required
  because a worker started from a web page belongs to that page and is restricted by
  x.com's content security policy; the model can only load on the extension's own origin.
* **`storage`** — remembers your on/off setting and the popup counters, as above.
* **Access to `x.com` / `twitter.com`** — the extension only runs on those sites, because
  that is the only place it has anything to do.

The extension requests no other permissions. It cannot read your other tabs, your
browsing history, your cookies, or your account credentials.

## Children

The extension is not directed at children and collects no data from anyone.

## Changes

Any future version that changes what data is handled will update this policy before
release, and any version that added a network request would require new permissions that
Chrome would prompt you to approve.

## Contact

Questions or reports: open an issue at
https://github.com/HappyWalkers/twitter-duplicate-filter/issues
