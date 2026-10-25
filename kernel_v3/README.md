# Version 3 (V3) — Neural-Forward Re-weighting + 3-Point Convex Shift Blend

## 1. Overview & Motivation
In Version 1 and 2, the 4-view/5-view LinearSVC carried a weight of $w = 1.0$, while the deep foundation models (Qwen3-14B, Qwen2.5-14B, DeBERTa-v3) were under-represented. Version 3 rebalances the ensemble hierarchy so that our 14-billion parameter deep neural backbones drive primary decision-making, while augmented stylometry provides complementary regularized signals.

## 2. Key Upgrades in V3

1. **Neural-Forward Weight Distribution:**
   - Elevates the top-performing deep foundation models:
     - **Qwen3-14B LoRA:** $w = 0.55$ (up from $0.40$), centred twin $w = 0.15$ (up from $0.10$).
     - **Qwen2.5-14B LoRA:** $w = 0.25$ (up from $0.20$), centred twin $w = 0.10$.
     - **DeBERTa-v3-large:** $w = 0.30$ (up from $0.25$).
     - **5-View LinearSVC:** Calibrated to $w = 0.75$ (down from $1.00$).
   - Total Neural Weight = $1.35$ vs Stylometric Weight = $0.75$ ($64\%$ neural dominance vs previous $45\%$).

2. **3-Point Convex Shift Combination:**
   - Evaluates three points along the topic debiasing trajectory ($\Delta = 0.60, 0.75, 0.90$) and computes a smooth probability mixture:
     $$P_{\text{final}} = 0.25 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.60\Delta)) + 0.50 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.75\Delta)) + 0.25 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.90\Delta))$$
   - Mitigates threshold flipping on the 18-class boundary for ambiguous prompt clusters.

3. **Retained V2 Innovations:**
   - 4,926 curated, balanced passages from `prvsiyan/passages_clean.csv`.
   - 5th stylometric feature stream: Punctuation cadence & rhythm n-grams.
   - Temperature scaling ($T = 1.25$) on LinearSVC logits.

## 3. Kaggle Configuration

- **Kernel ID:** `sanjay0614/ntl-v3-neural-reweighted`
- **Accelerator:** 2x NVIDIA T4 GPU
- **Internet:** Off
- **Datasets:** `nishantdahal/ntl-handwriting-blend`, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
- **Models:** `qwen-lm/qwen-3/Transformers/14b-base/1`, `qwen-lm/qwen2.5/Transformers/14b/1`
