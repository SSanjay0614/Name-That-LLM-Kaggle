# Version 4 (V4) — Soft Typographic Prior Gating + Calibrated Neural Blend

## 1. Overview & Motivation
In previous public experiments (e.g. Nuridudak), "Typography Guards" attempted hard rule enforcement (zeroing probabilities of entire families if a mark was present or absent). This proved brittle on the private leaderboard because real LLM generations occasionally include stray quotes or irregular spacing. Version 4 introduces **Soft Typographic Prior Gating**: continuous, uncertainty-scaled logit adjustments that gently guide ambiguous decisions without ever hard-zeroing predictions.

## 2. Key Upgrades in V4

1. **Soft Typographic Prior Gating:**
   - Based on empirical provenance statistics:
     - Anthropic (`claude`) and Zhipu AI (`glm`) models exhibit near-$0.0\%$ curly quotation/apostrophe usage, but have high em-dash frequencies ($>1.2$ per 100 words).
     - OpenAI (`gpt-5.x`, `gpt-6`) and Google (`gemini-3.7/3.8`) models have high typographic apostrophe shares ($>0.85$).
   - Rather than binary filtering, logit offsets are scaled by the row's uncertainty:
     $$\text{uncertainty} = \text{clip}\left(\frac{2.0 - \text{margin}}{2.0}, 0.0, 1.0\right)$$
   - Only ambiguous rows with small confidence margins receive soft typographic priors.

2. **Calibrated Neural-Forward Blending:**
   - Qwen3-14B LoRA ($w = 0.55$, twin $w = 0.15$)
   - Qwen2.5-14B LoRA ($w = 0.25$, twin $w = 0.10$)
   - DeBERTa-v3-large ($w = 0.30$)
   - Calibrated 5-View LinearSVC ($w = 0.75$, $T = 1.25$) on 35k pool + 4,926 curated passages.

3. **3-Point Convex Topic Shift Mixture:**
   - Evaluated across shift strengths $0.65$, $0.75$, and $0.85$ on the typographically regularized scores:
     $$P_{\text{final}} = 0.30 \cdot \text{Sinkhorn}(\text{softmax}(S_{\text{typo}} + 0.65\Delta)) + 0.50 \cdot \text{Sinkhorn}(\text{softmax}(S_{\text{typo}} + 0.75\Delta)) + 0.20 \cdot \text{Sinkhorn}(\text{softmax}(S_{\text{typo}} + 0.85\Delta))$$

## 3. Kaggle Configuration

- **Kernel ID:** `sanjay0614/ntl-v4-soft-typo-priors`
- **Accelerator:** 2x NVIDIA T4 GPU
- **Internet:** Disabled
- **Attached Datasets:** `nishantdahal/ntl-handwriting-blend`, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
- **Attached Models:** `qwen-lm/qwen-3/Transformers/14b-base/1`, `qwen-lm/qwen2.5/Transformers/14b/1`
