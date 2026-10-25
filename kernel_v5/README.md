# Version 5 (V5) — Multi-Scale Centred Twins + Soft Sibling Typographic Priors + Calibrated Shift Mixture

## 1. Overview & Analysis of V1 (0.621)
Analysis of our Version 1 baseline (which achieved **0.621**) and original public experiment logs revealed:
1. **Over-correction by Shift Multplier**: V1 applied `strength = 0.82` for the grouped SVC shift. Historical ablations indicate that while a conservative shift (`0.50 - 0.65`) achieves **0.628+**, pushing past $0.80$ over-corrects on borderline classes and causes regression.
2. **SVC Weight Dominance**: In V1, LinearSVC carried $w = 1.0$, which contributed ~50% of the entire ensemble and overpowered the 14-billion parameter Qwen foundations on complex prompts.
3. **Identical Sibling Bottlenecks**: `glm-5.3` vs `glm-5.3-flash` and `gemini-3.7-flash` vs `gemini-3.8-flash` are difficult to resolve using raw embeddings alone.

## 2. Key Upgrades in V5

1. **Multi-Scale Centred Twin Debiasing:**
   - Instead of a single flat clustering cutoff, V5 computes two levels of prompt hierarchy:
     - **Tight prompt clusters** ($d \le 0.82$): Identical prompts across multiple models.
     - **Broad topic clusters** ($d \le 0.88$): Paraphrased or domain-related prompts.
   - Computes multi-scale residual mean subtraction:
     $$Z_c = Z - 0.70 \cdot \bar{Z}_{\text{tight}} - 0.30 \cdot \bar{Z}_{\text{broad}}$$
   - Completely decouples prompt topic from stylistic attribution.

2. **Calibrated Shift Mixture in the Confirmed Sweet Spot ($0.50 - 0.72$):**
   - Blends three probability distributions across the conservative optimal range:
     $$P_{\text{final}} = 0.30 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.50\Delta)) + 0.50 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.62\Delta)) + 0.20 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.72\Delta))$$
   - Eliminates the over-correction issue observed with 0.82 in V1.

3. **Soft Typographic & Formatting Prior Gating:**
   - Invariant formatting signals: Claude and GLM models almost never output curly quotes/apostrophes (`’`, `“`, `”`), but frequently output em-dashes (`—`).
   - Gated by prediction uncertainty: only ambiguous rows receive gentle soft logit adjustments ($0.50 - 0.80 \cdot u$).

4. **Neural-Forward Weighting:**
   - Qwen3-14B: $w = 0.55$, twin $w = 0.15$
   - Qwen2.5-14B: $w = 0.25$, twin $w = 0.10$
   - DeBERTa-v3: $w = 0.30$
   - 5-View Stylometric LinearSVC: $w = 0.75$ ($T = 1.25$) trained on 35k pool + 4,926 curated passages.

## 3. Kaggle Configuration

- **Kernel ID:** `sanjay0614/ntl-v5-multiscale-twins`
- **Accelerator:** 2x NVIDIA T4 GPU
- **Internet:** Disabled
- **Attached Datasets:** `nishantdahal/ntl-handwriting-blend`, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
- **Attached Models:** `qwen-lm/qwen-3/Transformers/14b-base/1`, `qwen-lm/qwen2.5/Transformers/14b/1`
