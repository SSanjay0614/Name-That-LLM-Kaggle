# Version 1 Baseline (V1) — Two-Stage LoRA + Adaptive Centred Twins + Topic Debiased SVC

## 1. Overview & Objective
This notebook deploys our first competitive baseline for the **Name That LLM** competition, based on the top public architecture scoring **~0.628** on the public leaderboard (Rank ~16-20 tier).

## 2. Core Architecture & Components

1. **4-View Stylometric LinearSVC ($w = 1.0$):**
   - Trained on the competition `train.csv` concatenated with the 35,000 public passage dataset (`pool_v4.parquet`).
   - Features:
     - Character 3-to-5 n-grams ($250\text{k}$ features, weight 1.0).
     - Word 1-to-3 n-grams ($180\text{k}$ features, weight 0.7).
     - Shape rhythm tokens ($60\text{k}$ features, weight 0.5): All letter sequences mapped to `X` (uppercase) or `x` (lowercase).
     - Small function words ($50\text{k}$ features, weight 0.5): Function words and punctuation preserved; all semantic content words mapped to `CONTENT`.

2. **Decoder LoRAs on Qwen Foundations:**
   - **Qwen3-14B LoRA ($w = 0.4$, twin $w = 0.1$):** Truncated to first 40 layers, evaluated across layers $[3, 5, 8, 11, 16, 21, 27, 32]$.
   - **Qwen2.5-14B LoRA ($w = 0.2$, twin $w = 0.1$):** Truncated to first 48 layers with identical multi-layer hidden state pooling.

3. **Centred Twin Debiasing (Prompt Clustering):**
   - Hierarchical agglomerative clustering (`fcluster` with average linkage at cosine distance $0.85$) groups test passages generated from identical prompts.
   - Mean embedding subtraction isolates the model author's stylistic signature from prompt/topic semantics.

4. **DeBERTa-v3-large Encoder ($w = 0.25$):**
   - Independent bidirectional transformer encoder fine-tuned on the author attribution objective with length-sorted dynamic batching.

5. **Grouped SVC Shift & Margin Gating (Romantamrazov Innovation):**
   - Calculates cluster-level mean shift on the SVC decision function: $\Delta = \bar{s} - s_{\text{cluster}}$.
   - Uncertainty gate modulates shift:
     $$\text{gate} = \text{clip}\left(\frac{2.0 - (\text{top}_1 - \text{top}_2)}{1.6}, 0.15, 1.0\right)$$
   - Only ambiguous predictions receive the debiasing adjustment, protecting high-confidence predictions.

6. **Sinkhorn Optimal Transport Balancing:**
   - 300 iterations of Sinkhorn matrix scaling enforce equal marginal probability mass across all 18 classes, optimizing directly for Balanced Accuracy.

## 3. Kaggle Dependencies & Resources

- **Kernel Type:** Kaggle Notebook
- **Accelerator:** GPU (2x NVIDIA T4)
- **Internet:** Off (`enable_internet: false`)
- **Dataset Attached:** `nishantdahal/ntl-handwriting-blend`
- **Model Hub Models Attached:**
  - `qwen-lm/qwen-3/Transformers/14b-base/1`
  - `qwen-lm/qwen2.5/Transformers/14b/1`

## 4. Submission & Fast Validation

- When `len(test) < 50` (interactive commit/save), the notebook outputs a fallback prediction within seconds to avoid wasting GPU quota.
- When submitted against the hidden evaluation test set, the full multi-GPU neural pipeline executes end-to-end.
