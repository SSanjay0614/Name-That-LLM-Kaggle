# NTL V23 ATLAS 20 Sharper Blend

- **Competition:** Name That LLM (`name-that-llm`)
- **Hypothesis:** Combines the sharper Qwen & Helper mix (`qwen=0.15, helper=0.15`) from LB 0.643 with the high-impact ATLAS 20% weighting (`atlas=0.20`) proven in top public kernels to boost balanced accuracy across low-margin frontier classes.
- **Weights:** `Variant 6: dict(qwen=0.15, helper=0.15, atlas=0.20)`
- **GPU Protection:**
  - Fast commit preview runs in ~25 seconds on visible test (0.00h GPU consumed).
  - Heavy inference executes only on hidden test set during competition scoring reruns.
- **Dependencies:** 100% public, verified datasets, kernels, and Qwen models.
- **Kaggle result:** submission `57831323` scored **0.640**.
