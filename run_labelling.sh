#!/usr/bin/env bash
# Run this tonight once self-play is done (or when you want to stop it and label).
set -e
cd "$(dirname "$0")"
PY="C:/Users/44783/AppData/Local/Programs/Python/Python312/python.exe"
SF="./stockfish/stockfish-windows-x86-64-universal.exe"
NODES=15000
WORKERS=8

echo "=== stopping self-play ==="
powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*multiprocessing.spawn*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" || true
echo "  waiting for the parent to merge its per-worker files..."
sleep 15

echo "=== combining position files ==="
cat nnue/sp_batch1.txt nnue/sp_cont_s5.txt nnue/sp_cont3.txt nnue/sp_cont3.txt.w* 2>/dev/null | sort -u > nnue/sp_all.txt || \
  cat nnue/sp_batch1.txt nnue/sp_cont_s5.txt nnue/sp_cont3.txt 2>/dev/null | sort -u > nnue/sp_all.txt
wc -l nnue/sp_all.txt

echo "=== Stockfish labelling: $NODES nodes/pos, $WORKERS workers ==="
SF_PATH="$SF" "$PY" -m nnue.sflabel nnue/sp_all.txt nnue/sp_labelled.txt "$NODES" "$WORKERS"
echo "=== DONE -> nnue/sp_labelled.txt ==="
wc -l nnue/sp_labelled.txt
