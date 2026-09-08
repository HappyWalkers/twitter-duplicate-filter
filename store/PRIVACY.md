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

Everything below lives in Chrome's local extension storage on your own computer. None of
it is ever transmitted.

* whether the extension is switched on, and the sensitivity setting;
* a small counter of how many posts have been seen and folded, shown in the popup;
* **a memory of posts you have already scrolled past**, so the same story is still
  recognised after you reload the page.

That last item deserves a plain description, because it is a record of your reading:

* It stores, per post, a **numeric fingerprint** of the post's text, the post's id, and
  the author's handle. It does **not** store the text itself, images, or links.
* The fingerprint is a 384-number vector compressed to 384 bytes. It is not reversible
  into the original post, but it is derived from it: given a candidate piece of text, it
  can be checked for a match. Treat it as a record of what you read, because that is
  what it is.
* It is capped at **4,000 posts** and each entry **expires after 7 days**.
* **You can erase it at any time** — click the extension icon and press *Forget
  remembered posts*. The count above the button tells you how many are held.

Post text itself is held **in memory only**, for at most the most recent 400 posts, and is
discarded when the tab is closed. It is never written to disk.

## Permissions, and why each is needed

* **`offscreen`** — creates a hidden extension page that hosts the model. Required
  because a worker started from a web page belongs to that page and is restricted by
  x.com's content security policy; the model can only load on the extension's own origin.
* **`storage`** — holds your settings, the popup counters, and the post fingerprints
  described above. All local; nothing is synced or transmitted.
* **Access to `x.com` / `twitter.com`** — the extension only runs on those sites, because
  that is the only place it has anything to do.

The extension requests no other permissions. It cannot read your other tabs, your
browsing history, your cookies, or your account credentials.

## Children

The extension is not directed at children and collects no data from anyone.

## Turning the memory off

There is no separate switch for it: switching the extension off stops it recording, and
*Forget remembered posts* erases what is held. If you would rather it never wrote
anything, use the extension with the toggle off, or uninstall it — uninstalling removes
the extension's storage with it.

## Changes

Any future version that changes what data is handled will update this policy before
release, and any version that added a network request would require new permissions that
Chrome would prompt you to approve.

## Contact

Questions or reports: open an issue at
https://github.com/HappyWalkers/twitter-duplicate-filter/issues
