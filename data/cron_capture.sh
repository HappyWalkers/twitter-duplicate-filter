#!/bin/bash
# Periodic corpus top-up for the dedup benchmark.
#
# Run on a schedule, this buys TEMPORAL diversity, which a single sitting cannot:
# X's trend list rotates every few hours, so N runs spread over a day yield N sets of
# genuinely different stories instead of a deeper sample of the same ones. It also
# keeps each run well under X's search rate limit, which is what capped the earlier
# one-shot sweeps at ~30 navigations.
#
# Self-healing on Chrome: --remote-debugging-port is IGNORED by Chrome on the default
# profile (hardening since ~v136), so this uses a dedicated cloned profile. If that
# browser is gone, it relaunches it before capturing.
set -uo pipefail

REPO="/home/yunpeng/x-timeline-dedup"
PROFILE="$HOME/.cache/x-dedup-profile"
PORT=9222
LOG="/tmp/cron_capture_$(date +%Y%m%d_%H%M).log"

cd "$REPO" || exit 1

ensure_chrome() {
  if curl -s --max-time 4 "http://127.0.0.1:$PORT/json/version" >/dev/null 2>&1; then
    echo "chrome: already listening on $PORT"
    return 0
  fi
  echo "chrome: not reachable, relaunching on cloned profile"
  # Refresh the session material from the live profile so cookies stay current.
  if [ -d "$HOME/.config/google-chrome/Default" ]; then
    for f in Cookies Cookies-journal Preferences "Local Storage" "Session Storage" Network; do
      cp -a "$HOME/.config/google-chrome/Default/$f" "$PROFILE/Default/" 2>/dev/null
    done
    cp -a "$HOME/.config/google-chrome/Local State" "$PROFILE/" 2>/dev/null
  fi
  nohup google-chrome --user-data-dir="$PROFILE" --remote-debugging-port=$PORT \
      --no-first-run --no-default-browser-check --disable-session-crashed-bubble \
      --restore-last-session=false "https://x.com/home" >/dev/null 2>&1 &
  for _ in $(seq 1 20); do
    sleep 2
    curl -s --max-time 4 "http://127.0.0.1:$PORT/json/version" >/dev/null 2>&1 && {
      echo "chrome: up"; return 0; }
  done
  echo "chrome: FAILED to come up"; return 1
}

BEFORE=$(wc -l < data/posts.jsonl 2>/dev/null || echo 0)

{
  echo "=== cron capture $(date -Is) ==="
  ensure_chrome || exit 1

  # Alternate between the trend sweep and a non-English query sweep. The corpus is
  # Latin-heavy because X's /explore trends are locale-pinned to the US, which left
  # only 2 Chinese posts -- not enough to test a multilingual model on Chinese at all.
  # RETUNED after measuring the benchmark's actual bottleneck.
  #
  # It is not post count. 4182 posts yielded only 53 macro-scoreable clusters, and a
  # single cluster supplied 50.5% of all positive pairs -- because pairs scale as
  # C(n,2), so deepening one story adds pairs to a cluster that already dominates while
  # adding no statistical power. The paired bootstrap could not separate the top two
  # models precisely because n_stories=53.
  #
  # So: MORE TRENDS, SHALLOWER CAP. 30 trends x 16 posts beats 16 trends x 30 for
  # distinct-story count at identical crawl cost, and flattens the concentration.
  #
  # The broad-topic multilingual queries are also retired: they returned ~119 stories
  # per 120 posts (essentially all singletons), because searching a topic word like
  # 日本 returns unrelated posts sharing a keyword, not one event. Replaced with
  # BREAKING-NEWS terms, which do concentrate on events and therefore form clusters.
  HOUR=$(date +%H)
  if [ $((10#$HOUR % 2)) -eq 0 ]; then
    echo "--- mode: wide trend sweep (max distinct stories) ---"
    "$REPO/.venv/bin/python" -u data/capture.py \
        --per-source 40 --per-trend 16 --trends 30 \
        --scrolls 10 --pause 1300 --nav-delay 4500
  else
    # ROTATE the term list. A FIXED list stops adding distinct stories after one pass:
    # measured +0 new trends on a repeat sweep, because the same 14 queries return the
    # same buckets and only deepen them -- which actively hurts, since concentration is
    # already the benchmark's biggest weakness. Rotating by hour keeps each sweep
    # landing on events the corpus has not seen.
    case $(( (10#$HOUR / 2) % 3 )) in
      0) Q="突发,快讯,刚刚宣布,地震,发布会,官宣,速報,発表,会見,긴급,속보,대통령,عاجل,انفجار" ;;
      1) Q="曝光,通报,声明,辟谣,最新消息,独自,判明,決定,단독,확정,بيان,تصريح,última hora,urgente" ;;
      2) Q="回应,道歉,宣布退出,正式启动,现场,一報,速報値,発生,긴급속보,공식입장,عاجل الآن,انفراد,ao vivo,alerta" ;;
    esac
    echo "--- mode: non-English event sweep (rotation $(( (10#$HOUR / 2) % 3 ))) ---"
    "$REPO/.venv/bin/python" -u data/capture.py \
        --per-trend 16 --scrolls 10 --pause 1300 --nav-delay 5000 --queries "$Q"
  fi
} >>"$LOG" 2>&1

AFTER=$(wc -l < data/posts.jsonl 2>/dev/null || echo 0)
echo "posts: $BEFORE -> $AFTER (+$((AFTER - BEFORE)))"
echo "log: $LOG"
tail -14 "$LOG"
