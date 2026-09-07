#!/usr/bin/env bash
# Keep agy labelling across quota windows without supervision.
#
# agy's individual quota yields roughly 1,300 labels before it refuses with
# "Individual quota reached ... Resets in Xh Ym", so the backlog needs several windows.
# label.py checkpoints every batch and --resume skips what is already done, which makes
# simply re-running it the whole recovery strategy: a quota stop costs at most the batch
# that was in flight.
#
# Sleeping ~2h40m after a quota stop (a little past the observed reset) rather than
# polling every few minutes -- a poll during a closed window still spends a real API call
# to be told no.
set -u
cd /home/yunpeng/x-timeline-dedup || exit 1
LOG=data/agy_run.log
MAX_ROUNDS=12

for round in $(seq 1 $MAX_ROUNDS); do
  before=$(.venv/bin/python -c "import json;print(len(json.load(open('data/labels_agy.json'))))" 2>/dev/null || echo 0)

  echo "=== resume round $round ($(date -Is)), $before labels so far ===" >> "$LOG"
  .venv/bin/python -u data/label.py --labeller agy --resume --exclude gold.json \
      --batch 40 --timeout 600 >> "$LOG" 2>&1

  after=$(.venv/bin/python -c "import json;print(len(json.load(open('data/labels_agy.json'))))" 2>/dev/null || echo 0)
  remaining=$(.venv/bin/python - <<'PY' 2>/dev/null || echo 1
import json
from pathlib import Path
posts=[json.loads(l) for l in Path('data/posts.jsonl').read_text().split('\n') if l.strip()]
gold=json.load(open('data/gold.json')); agy=json.load(open('data/labels_agy.json'))
print(sum(1 for p in posts if p['statusId'] not in gold and p['statusId'] not in agy))
PY
)
  echo "=== round $round done: $before -> $after labels, $remaining backlog posts left ===" >> "$LOG"
  [ "$remaining" -eq 0 ] && { echo "=== backlog complete ===" >> "$LOG"; break; }

  if tail -40 "$LOG" | grep -q "quota"; then
    sleep 9600          # past the observed ~2h34m reset
  elif [ "$after" -le "$before" ]; then
    # No progress and no quota message: something else is wrong. Back off rather than
    # spin, and let the round cap end it.
    sleep 1800
  else
    sleep 60
  fi
done
