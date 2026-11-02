# NTL V22 Sharper Blend 0.643

- **Competition:** Name That LLM (`name-that-llm`)
- **Base:** `areebbb/0-643-same-stack-sharper-blend-enjoy` (Public LB **0.643**, SOTA)
- **Architecture:**
  - Fold-averaged two-stage LoRA24 (Qwen3-8B-Base, Qwen3-14B-Base, Qwen2.5-7B)
  - 4-view n-gram SVM on train+pool + shared-prompt Gaussian heads + adaptive centered twins
  - ATLAS source-conditioned likelihood
  - Variant 4 re-blend (`qwen=0.15, helper=0.15, atlas=0.10`) + Sinkhorn balancing
- **GPU Protection & Fast Commit:**
  - Fast commit preview runs in ~20 seconds on visible test (0.00h GPU consumed).
  - Heavy inference (2h) executes only on the hidden test set during competition scoring reruns.
- **Dependencies:** 100% public, verified datasets, kernels, and Qwen models.
- **Kaggle result:** submission `57830437` scored **0.643**.
