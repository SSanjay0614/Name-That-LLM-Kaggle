# V14 — Name That LLM V14 Quality SVC Pool

Replace V10's 75,783-row SVC fit pool with the 40,129-row curated public-only SVC pool. Its published prompt-grouped, SVC-only five-fold validation reported balanced accuracy 0.4582 versus 0.4430 for the original pool. This is component OOF evidence, not full-ensemble or leaderboard evidence.

This is a controlled branch from V10. All neural members, the official-only prompt-residual readout, blend weights, and Sinkhorn balancing remain unchanged; only the four-view n-gram SVC training pool changes.

Dataset attribution: fleongg/NTL Quality Selected Public SVC Pool; its per-row source and license catalog is retained in the attached dataset. The attached dataset provides per-row source/license provenance. No corpus is copied into this repository. A visible three-row Kaggle smoke run checks execution/format only; the full hidden-test rerun determines leaderboard performance.
