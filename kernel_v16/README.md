# V16 — Grouped SVC source ablation (validation only)

Runs five prompt-grouped folds on Kaggle for four SVC data recipes: V10/V13 base pool, base plus GLM only, base plus Kimi only, and base plus both. Records pooled balanced accuracy and per-label recall in `/kaggle/working/v16_svc_source_ablation.json`.

This is a component diagnostic, not full-ensemble OOF and not a competition submission. It cannot pass the action plan's submission gate on its own. The actual feature fitting runs in Kaggle; the local builder only writes notebook and metadata files.

Baseline pool: Nishant Dahal, [NTL two-stage LoRA part two](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend), CC BY-SA 4.0 (the pool includes ShareAlike material). Added data: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0. Prompt bank attribution remains with Seyit Kaan Güneş, [NTL Extra API Passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages), CC BY 4.0.
