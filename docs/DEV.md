# Working notes

Our build on top of the starter. The goal: a Leela-style ResNet (policy + value)
driven by a batched PUCT search, plus an opening book and Syzygy tablebases.
Target ~2200-2500 Elo. Upload closes 2026-09-11 11:00.

## Layout

| Path | Ships in zip? | What |
|---|---|---|
| `agent.py` | yes | entrypoint; search + time management |
| `encoding.py` | yes | FEN <-> (19,8,8) planes, move <-> policy index (AlphaZero 8x8x73) |
| `weights/` | yes | the exported `.onnx` model |
| `harness/` | no | the platform's protocol + clock. **Do not edit.** Linux only. |
| `tools/localarena.py` | no | Windows-friendly in-process arena with an Elo estimate |
| `training/` | no | model definition, data pipeline, train loop; runs on Kaggle/cloud |
| `tests/` | no | encoding round-trip and invariant tests |

`encoding.py` and `training/model.py` must agree on `N_PLANES` (19) and
`POLICY_SIZE` (4672). The policy head emits logits in `from_square * 73 + plane`
order so `policy.reshape(-1)` lines up with `encoding.move_to_index`.

## Running things

Local dev is on Windows; the platform and CI are Linux.

```
py -3.12 -m pytest                                   # encoding tests
py -3.12 -m tools.localarena --opponent baselines/greedy --games 200 --workers 8
py -3.12 -m training.model                           # print net configs + param counts
```

`make play` / `make arena` / `make gate` use the real `harness/` and only work on
Linux (CI runs them). `tools/localarena.py` mirrors `harness/referee.py`'s rules,
clock and 300-ply adjudication but not process isolation or the output cap; use
the real harness on Linux before spending upload quota (6/day).

## Training

`pip install -r training/requirements.txt` on the GPU box. Not the user's RX 6600
(weak ROCm support) - use Kaggle (free 30 GPU-hr/week P100/T4).

```
# 1. raw dumps -> shards  (Lichess eval DB is Stockfish-annotated; distillation is allowed)
py -3.12 -m training.prepare eval --input data/raw/lichess_db_eval.jsonl.zst \
    --out data/shards/eval --limit 8000000
py -3.12 -m training.prepare pgn  --input data/raw/lichess_elite.pgn.zst \
    --out data/shards/games --min-elo 2300 --limit 6000000

# 2. train (Kaggle GPU). writes weights/model.pt + weights/model.onnx each epoch
py -3.12 -m training.train --shards data/shards/eval data/shards/games \
    --config medium --epochs 3 --batch 2048 --out weights

# 3. export / quantize a checkpoint
py -3.12 -m training.export --checkpoint weights/model.pt --out weights/model.onnx --quantize
py -3.12 -m training.export --random --config tiny --out weights/model.onnx   # plumbing only
```

Modules: `training/labels.py` (numpy-free value maths, CI-tested), `training/pack.py`
(36-byte position records), `training/prepare.py` (raw -> shards, CI-tested),
`training/data.py` (torch Dataset), `training/model.py` (`ChessNet`),
`training/train.py`, `training/export.py` (ONNX via `dynamo=True`, single file).

## Runtime path (shipped)

`agent.py` -> `inference.Evaluator` (onnxruntime, 1 core) -> `encoding`. Current
`agent.py` is a **1-ply value look-ahead**, no tree search yet - it exists to
measure the raw net and prove the pipe. PUCT lands on top of the same `Evaluator`.

`weights/` is gitignored during dev; `make zip` reads `weights/model.onnx` off disk.

## Rules that bite

- 1 core, 2 GB, no net/GPU at runtime. `torch.set_num_threads(1)` - but we ship
  onnxruntime, so set `intra_op_num_threads=1` on the session.
- Read-only FS except 256 MB `/tmp` (HOME + caches already point there).
- Zip is first on `sys.path`: never shadow a stdlib/dep module name.
- No native binaries in the zip; compiled deps from PyPI wheels only.
- Flagging is the most common self-inflicted loss. Always keep a legal fallback
  move ready and return best-so-far when the time budget is gone.
