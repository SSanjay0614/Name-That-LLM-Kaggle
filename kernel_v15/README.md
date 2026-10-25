# V15 — Name That LLM V15 Quality GLM Kimi SVC Pool

Combine V14's curated public-only SVC pool with 1,516 GLM/Kimi passages. Exact text overlap between these two corpora was checked locally and was zero. This tests whether the separately observed curated-pool OOF gain and GLM/Kimi leaderboard gain are complementary; the combined effect is unvalidated.

This is a controlled branch from V10. All neural members, the official-only prompt-residual readout, blend weights, and Sinkhorn balancing remain unchanged; only the four-view n-gram SVC training pool changes.

Dataset attribution: fleongg/NTL Quality Selected Public SVC Pool; its per-row source and license catalog is retained in the attached dataset. The additional OpenRouter passages are from Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. The attached dataset provides per-row source/license provenance. No corpus is copied into this repository. A visible three-row Kaggle smoke run checks execution/format only; the full hidden-test rerun determines leaderboard performance.
