# NTL V24 Qwen20 Blend

- **Competition:** Name That LLM (`name-that-llm`)
- **Hypothesis:** Variant 2: `dict(qwen=0.20, helper=0.10, atlas=0.10)`. Increases the influence of the 27-billion parameter Qwen3.8-27B fulltext model on selected difficult rows from 15% to 20%.
- **GPU Protection:**
  - Fast commit preview runs in ~25s on visible test (0.00h GPU consumed).
  - Heavy inference executes only on hidden test set during competition scoring reruns.
- **Dependencies:** 100% public, verified datasets, kernels, and Qwen models.
- **Kaggle result:** submission `57831715` scored **0.640**.
