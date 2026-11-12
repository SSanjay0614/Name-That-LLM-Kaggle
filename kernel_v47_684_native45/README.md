# NTL v47 684 native45

- Kaggle kernel: `sanjay0614/ntl-v47-684-native45` (private, GPU, no internet)
- Source: `prvsiyan/the-prompt-is-not-the-author` v37 (public, Apache-2.0)
- Built by: `tools_build_684.py` (2026-11-12)

One setting away from the 0.684 code: the native Qwen3-Embedding expert share moves 0.40 -> 0.45 (the author's v33 test), with the Sinkhorn strength restored to the 0.684 value so the two readings stay separable.

## Edits

- `'--sinkhorn', '0.20'` -> `'--sinkhorn', '0.30'`  (restore the 0.684 Sinkhorn strength so only one setting differs from 0.684)
- `'--native_share', '0.40'` -> `'--native_share', '0.45'`  (the v33 one-argument test: native Qwen3-Embedding expert share 0.40 -> 0.45)

## Attachments

- 17 datasets, 8 kernel outputs, 8 models - exactly the author's own lists, read from the pull API rather than assumed.
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`
- Competition: `name-that-llm`

## Contract

- Output: `submission.parquet` with `id, model`; the hidden scoring rerun executes the full two-GPU pipeline.
- The commit run must hit the notebook's own public-fixture preview path before the slot is spent.
