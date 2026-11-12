# NTL v44 684 sinkhorn20

- Kaggle kernel: `sanjay0614/ntl-v44-684-sinkhorn20` (private, GPU, no internet)
- Source: `prvsiyan/the-prompt-is-not-the-author` v37 (public, Apache-2.0)
- Built by: `tools_build_684.py` (2026-11-12)

The author's current version is itself the one-argument test: Sinkhorn balancing strength 0.30 -> 0.20, everything else the 0.684 code. Forum reports put the gain at about +0.003 because the hidden class mix is uneven and strong balancing over-corrects.

## Edits

- none (verbatim copy)

## Attachments

- 17 datasets, 8 kernel outputs, 8 models - exactly the author's own lists, read from the pull API rather than assumed.
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`
- Competition: `name-that-llm`

## Contract

- Output: `submission.parquet` with `id, model`; the hidden scoring rerun executes the full two-GPU pipeline.
- The commit run must hit the notebook's own public-fixture preview path before the slot is spent.
