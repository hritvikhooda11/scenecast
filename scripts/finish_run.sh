#!/bin/bash
# Wait for the three AV2 runs to finish, then evaluate everything and render every figure.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNS="base no-map no-neighbors"
EPOCHS=${EPOCHS:-40}

for _ in $(seq 1 2000); do
    done_count=0
    for n in $RUNS; do
        c=$(tr '\r' '\n' < "logs/$n.log" 2>/dev/null | grep -cE "^epoch +[0-9]+ \|")
        [ "${c:-0}" -ge "$EPOCHS" ] && done_count=$((done_count + 1))
    done
    [ "$done_count" -eq 3 ] && break
    sleep 30
done

echo "=== training finished ==="
for n in $RUNS; do echo "--- $n"; tr '\r' '\n' < "logs/$n.log" | grep -E "^epoch +[0-9]+ \|" | tail -1; done

echo "=== evaluation ==="
$PY -m scenecast.evaluate --data data/processed/av2/val \
    --ckpt runs/base/best.pt runs/no-map/best.pt runs/no-neighbors/best.pt \
    --out runs/results.json 2>&1 | tail -8

echo "=== figures ==="
$PY -m scenecast.viz --data data/processed/av2/val --ckpt runs/base/best.pt --out assets/grid.png 2>&1 | tail -1
$PY -m scenecast.viz --data data/processed/av2/val --ckpt runs/base/best.pt --out assets/failures.png --worst 2>&1 | tail -1
$PY -m scenecast.viz --data data/processed/av2/val --ckpt runs/base/best.pt --out assets/demo.gif --index 7 2>&1 | tail -1
$PY -m scenecast.curves --runs runs/base runs/no-map runs/no-neighbors --out assets/curves.png 2>&1 | tail -1
echo "=== all done ==="
