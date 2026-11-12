# NTL v46 684 punct

- Kaggle kernel: `sanjay0614/ntl-v46-684-punct` (private, GPU, no internet)
- Source: `kirilltym/lora-trained27-evidence-for-close-calls` v19 (public, Apache-2.0)
- Built by: `tools_build_684.py` (2026-11-12)

Author's version 19: the 0.684 ensemble plus a punctuation prior fitted on the official training answers (spaced em dash, em dash, curly quotes, curly apostrophe; a model with >=80 answers and zero occurrences of a pattern present in the query loses 0.70 log score). Prompt-held-out checks added 4-10 correct answers out of 2,159.

## Edits

- none (verbatim copy)

## Attachments

- 17 datasets, 8 kernel outputs, 8 models - exactly the author's own lists, read from the pull API rather than assumed.
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`
- Competition: `name-that-llm`

## Contract

- Output: `submission.parquet` with `id, model`; the hidden scoring rerun executes the full two-GPU pipeline.
- The commit run must hit the notebook's own public-fixture preview path before the slot is spent.
