#!/usr/bin/env bash
# Run this tonight once self-play is done (or when you want to stop it and label).
set -e
cd "$(dirname "$0")"
PY="C:/Users/44783/AppData/Local/Programs/Python/Python312/python.exe"
SF="./stockfish/stockfish-windows-x86-64-universal.exe"
# ~7M positions in <6h. Timed at 71 pos/s/worker @ 10k nodes even under load;
# 12k on the free machine -> ~4-5h for 7M. sflabel flushes every 2000/worker so
# a partial run is fully usable if you cut it early.
NODES=12000
WORKERS=10

echo "=== stopping self-play ==="
powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*multiprocessing.spawn*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" || true
echo "  waiting for the parent to merge its per-worker files..."
sleep 15

echo "=== combining position files ==="
cat nnue/sp_batch1.txt nnue/sp_cont_s5.txt nnue/sp_cont3.txt nnue/sp_cont3.txt.w* > nnue/sp_all.txt 2>/dev/null
wc -l nnue/sp_all.txt

echo "=== Stockfish labelling: $NODES nodes/pos, $WORKERS workers ==="
SF_PATH="$SF" "$PY" -m nnue.sflabel nnue/sp_all.txt nnue/sp_labelled.txt "$NODES" "$WORKERS"
echo "=== DONE -> nnue/sp_labelled.txt ==="
wc -l nnue/sp_labelled.txt
