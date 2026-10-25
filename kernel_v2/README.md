# Version 2 (V2) — Augmented Stylometry + Punctuation Cadence + Dual-Strength Shift Ensembling

## 1. Overview & Motivation
Version 2 builds directly upon our validated Version 1 baseline (LB ~0.628). It addresses the primary failure mode of stylometric classification: topic contamination and unbalanced boundary confidence.

## 2. Key Upgrades in V2

1. **Curated Clean Corpus Integration (`prvsiyan/passages_clean.csv`):**
   - Attached `prvsiyan/prvsiyans-extended-name-that-llm-dataset`.
   - Injected **4,926 verified, balanced, leak-free passages** across all 18 models (245 to 288 per class) into the LinearSVC training set alongside `pool_v4.parquet` and `train.csv`.
   - Provides clean discrimination for challenging classes such as `claude-fable-5`, `gemini-3.7-flash`, `glm-5.3-flash`, and `kimi-k3`.

2. **5th Stylometric Feature View: Punctuation Rhythm & Spacing (`punct`):**
   - Extracts character 2-to-4 ngrams specifically from punctuation and whitespace:
     ```python
     def punct_text(t):
         return "".join(re.findall(r"[\s\.,;:!?\-\u2014\u2013\u2018\u2019\u201c\u201d\"\'\(\)\[\]\{\}/\\#*`~]", str(t)))
     ```
   - Captures authorial pacing, em-dashes (`\u2014`), quotation style (curly vs straight), semicolons, and parentheses spacing patterns that are completely invariant to semantic topic.

3. **Calibrated Temperature Scaling for LinearSVC ($T = 1.25$):**
   - Standard LinearSVC decision functions produce uncalibrated, peaky logits.
   - Temperature scaling:
     $$P_{\text{svc}} = \text{softmax}\left(\frac{\text{decision\_function}}{1.25}\right)$$
   - Prevents the SVM from overriding nuanced neural representations on complex or ambiguous passages.

4. **Dual-Strength Ensembled Topic Debiasing:**
   - Instead of relying on a single discrete shift multiplier, V2 computes a probability blend across conservative ($0.65$) and strong ($0.80$) debiased distributions:
     $$P_{\text{final}} = 0.40 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.65\Delta)) + 0.60 \cdot \text{Sinkhorn}(\text{softmax}(S + 0.80\Delta))$$
   - Attenuated by the calibrated margin gate:
     $$\text{gate} = \text{clip}\left(\frac{1.9 - \text{margin}}{1.5}, 0.12, 1.0\right)$$
   - Smooths the decision boundary and reduces discrete argmax variance on borderline test rows.

## 3. Kaggle Configuration

- **Kernel ID:** `sanjay0614/ntl-v2-augmented-stylometry`
- **Accelerator:** 2x NVIDIA T4 GPU
- **Internet:** Disabled
- **Attached Datasets:**
  1. `nishantdahal/ntl-handwriting-blend`
  2. `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
- **Attached Models:**
  1. `qwen-lm/qwen-3/Transformers/14b-base/1`
  2. `qwen-lm/qwen2.5/Transformers/14b/1`
