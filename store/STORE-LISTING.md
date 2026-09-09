# Chrome Web Store submission

Everything the submission form asks for. Copy the fields verbatim; the justification
wording matters, because vague answers are the most common cause of a rejection round-trip.

---

## Listing

**Name**  `Timeline Dedup for X`

**Summary** (132 char max, currently 118)
`Collapses posts about the same story into one. Runs entirely on your device — nothing is ever uploaded.`

**Category**  Social & Communication
**Language**  English

**Description**

> When a story breaks, every account you follow posts about it, and the same news reaches
> you ten times in different words.
>
> Timeline Dedup groups posts that are about the same story and folds the extras behind a
> "+N similar posts" control. Nothing is deleted and nothing is hidden for good — one
> click expands any group.
>
> **It compares posts to each other.** Blocklists and keyword filters have to know in
> advance what they are hiding. This does not: it reads the posts already on your screen
> and notices when several are saying the same thing, including across languages — a post
> in Japanese and a post in English about the same event are recognised as one story.
>
> **It runs entirely on your device.** The language model ships inside the extension.
> There are no servers, no accounts, no analytics, and no network permissions at all —
> the extension is not able to contact any website, and you can verify that in DevTools.
>
> To keep recognising a story after you reload, it remembers a compressed fingerprint of
> posts you have scrolled past — never the text itself. Entries expire after a week, are
> stored only on your own computer, and can be erased from the popup at any time.
>
> **It remembers across reloads.** Once you have been shown a story, posts about it are
> collapsed behind a "seen before" control next time — including the same post served to
> you again days later. Memory is kept on your device for a week and can be erased from
> the popup at any time.
>
> **It is tuned to under-fold rather than over-fold.** A missed duplicate costs you one
> redundant post. A wrong fold hides something you wanted and you would never know. The
> default folds only when it is right about 9 times in 10, measured on 14,201
> hand-labelled posts, and a Sensitivity control in the popup lets you trade that for
> more folding if you prefer.
>
> **Nothing but the model decides.** There are no keyword lists, no blocklists, and no
> rules about images, authors or post length. Every fold is one decision: are these two
> posts about the same thing?
>
> **It earns its keep during big news, not on a quiet timeline.** When one story is
> everywhere, roughly a third of what you scroll past is a repeat and the folding is
> obvious. On a calm Following feed there is simply less to fold — expect it to be
> quiet, and turn Sensitivity up if you would rather it acted more often.
>
> **It plays well with others.** It changes nothing about your timeline besides
> collapsing duplicates, and cooperates with Control Panel for Twitter if you use it.
>
> Open source: https://github.com/HappyWalkers/twitter-duplicate-filter

---

## Privacy practices tab

**Single purpose**
> Collapse posts on x.com that are about the same news story, so the same item is not
> shown repeatedly.

**Permission justifications** — answer each in terms of what breaks without it.

| Permission | Justification to paste |
|---|---|
| `offscreen` | Creates a hidden extension page that hosts the machine-learning model used to compare posts. Required because a worker created from a content script belongs to the x.com origin and is governed by that site's content security policy, which prevents the model's WebAssembly runtime from initialising. The offscreen document runs on the extension's own origin, where it can. |
| `storage` | Stores the user's on/off preference, sensitivity setting, two popup counters, and a local IndexedDB cache of numeric fingerprints of recently-seen posts so duplicates are still recognised after a page reload. The fingerprints are not the post text, entries expire after 7 days, and the user can erase them from the popup. Nothing is transmitted or synced. |
| Host access to `x.com`, `twitter.com` | The extension's entire function is to collapse duplicate posts on the user's timeline, so it must read post text on those sites. It runs nowhere else. |
| Remote code | **No.** All code ships in the package. The model file is data, not executable code, and it is bundled rather than downloaded — the extension declares no host permissions and makes no network requests. |

**Data usage** — you do collect nothing remotely, but be precise rather than clever here,
because an inaccurate disclosure is the kind of thing that gets an extension pulled later:

* Tick **"Website content"** and select **stored locally, not transmitted** if the form
  offers that distinction. The extension derives a fingerprint from post text and keeps it
  on the user's own machine for up to 7 days.
* Tick **none** of the transmission-based categories — nothing leaves the device, and the
  extension has no host permissions with which it could.
* Confirm all three certifications (no sale, no unrelated use, no creditworthiness use).

If a reviewer asks: post text is used in memory to compute similarity and discarded; only
an irreversible 384-byte numeric fingerprint, the post id and the author handle are cached
locally, expiring after 7 days, and erasable from the popup.

**Privacy policy URL**
`https://github.com/HappyWalkers/twitter-duplicate-filter/blob/main/store/PRIVACY.md`

---

## Assets

| Asset | Requirement | Status |
|---|---|---|
| Store icon | 128×128 PNG | `store/icons/icon128.png` |
| Small promo tile | 440×280 PNG | `store/store-assets/promo-440x280.png` |
| Screenshots | 1–5, **1280×800** or 640×400 | **you must capture these** — see below |

Screenshots are the one thing that cannot be generated from the repo, because they have
to show real folded posts. Capture them with:

```
python data/screenshot.py            # writes store/store-assets/screenshot-*.png
```

Take at least one showing a collapsed group with its "+N similar posts" chip, and ideally
one with the group expanded, so a reviewer can see that nothing is destroyed.

---

## Before you submit

1. **Set expectations in the listing, not in support email.** On a quiet Following feed
   the model folds very little -- measured well under 1% of posts -- because a personal
   timeline genuinely repeats itself less than a trending page does. Most of what a user
   notices day to day will come from the cross-session memory ("seen before") rather than
   from two posts on one screen. The description above says this deliberately: someone who
   installs expecting constant folding and sees none leaves a one-star review, and that is
   far more expensive than the honesty.
2. **Package size is 116MB zipped.** Under the 2GB limit, but expect a slower review than
   a small extension, and a visible download for users. Almost all of it is the model.
3. **Register as a developer** ($5 one-time) at
   https://chrome.google.com/webstore/devconsole
4. **Upload `dist-store.zip`** (produced by `./scripts/build-store.sh`).
5. **Expect the remote-code question.** The listing already answers it, but if a reviewer
   asks, the short version is: the ONNX file is model weights read by a WebAssembly
   runtime that ships in the package, there is no host permission, and no request leaves
   the extension.
6. **First review is typically a few days**; updates are usually faster.
