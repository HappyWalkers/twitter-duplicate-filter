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
> **It is tuned to under-fold rather than over-fold.** A missed duplicate costs you one
> redundant post. A wrong fold hides something you wanted and you would never know. The
> default folds only when it is right about 19 times in 20, measured on 14,201
> hand-labelled posts, and a Sensitivity control in the popup lets you trade that for
> more folding if you prefer.
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
| `storage` | Stores the user's on/off preference and two counters shown in the popup (posts seen, posts folded). No personal data, and nothing is transmitted. |
| Host access to `x.com`, `twitter.com` | The extension's entire function is to collapse duplicate posts on the user's timeline, so it must read post text on those sites. It runs nowhere else. |
| Remote code | **No.** All code ships in the package. The model file is data, not executable code, and it is bundled rather than downloaded — the extension declares no host permissions and makes no network requests. |

**Data usage** — tick **none** of the collection categories, and confirm all three
certifications (no sale, no unrelated use, no creditworthiness use). If asked to describe
handling of "website content": it is read in memory to compute similarity and discarded;
it is never stored or transmitted.

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

1. **Set expectations in the listing, not in support email.** The measured fold rate on
   a home timeline is ~0.3% of posts at the default and ~0.9% at the most eager setting,
   against ~2% across the mixed corpus and far more during breaking news. The description
   above says so deliberately: a user who installs expecting constant folding and sees
   none will leave a one-star review, and that is much more expensive than the honesty.
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
