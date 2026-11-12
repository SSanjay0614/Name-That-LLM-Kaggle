# NTL v45 684 targetmix50

- Kaggle kernel: `sanjay0614/ntl-v45-684-targetmix50` (private, GPU, no internet)
- Source: `heliosli/name-that-llm-lv683-a15` v14 (public, Apache-2.0)
- Built by: `tools_build_684.py` (2026-11-12)

The author's version 14 moves the Sinkhorn target mix 0.35 -> 0.50 (further toward the uniform training prior) on top of the 0.684 copy, and version 13 of the same notebook reproduced 0.684 exactly, so this is a calibrated base plus one documented knob.

## Edits

- none (verbatim copy)

## Attachments

- 17 datasets, 8 kernel outputs, 8 models - exactly the author's own lists, read from the pull API rather than assumed.
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`
- Competition: `name-that-llm`

## Contract

- Output: `submission.parquet` with `id, model`; the hidden scoring rerun executes the full two-GPU pipeline.
- The commit run must hit the notebook's own public-fixture preview path before the slot is spent.
