# V6 — Public Complement Reproduction

This notebook reproduces the inference pipeline from [Fleongg's public 0.636 submission](https://www.kaggle.com/code/fleongg/ntl-part-two-complement). Our competition submission **57591353** completed with a public score of **0.636**, matching the source notebook's reported score.

The recipe combines the two-stage LoRA main ensemble with a 15% complementary stack and a 10% ATLAS contribution on uncertain rows, then applies Sinkhorn balancing. It requires the datasets, model files, and source kernels listed in `kernel-metadata.json`.

## Result and validation scope

- All attached dataset and source-kernel references were accessible to the configured Kaggle account when checked.
- All notebook code cells pass Python syntax compilation.
- The competition submission ran the full hidden-test inference and scored **0.636**.
- The separate Kaggle smoke run used the three-row public preview and verified all required complementary members and ATLAS execution. It is an execution/packaging check, not an accuracy evaluation.
- The source author's fixed complementary-stack + ATLAS mixture does not have a reported full-ensemble out-of-fold ablation; the public score establishes the result but does not identify the individual component gains.
- The source notebook and its original credits are preserved unchanged.

V6 is the current best-scoring local candidate.
