# Name That LLM — Kaggle Competition

**Prize:** $50,000 USD  
**Competition Type:** Code Competition (Offline inference, 2x NVIDIA T4 GPU)  
**Evaluation Metric:** **Balanced Accuracy** across 18 candidate LLM classes  
**Deadline:** January 14, 2027  

---

## 1. Problem Statement & Objective

The objective is **AI provenance and authorship attribution**: given a short, unlabeled text passage, accurately predict which of **18 frontier large language models** generated it.

### The 18 Candidate LLMs (`LABELS`):
1. **OpenAI (5):** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
2. **Google (4):** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
3. **Anthropic (5):** `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`
4. **Alibaba (1):** `qwen3.8-max`
5. **Zhipu AI (2):** `glm-5.3`, `glm-5.3-flash`
6. **Moonshot AI (1):** `kimi-k3`

---

## 2. Evaluation Metric: Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{18} \sum_{c=1}^{18} \text{Recall}_c = \frac{1}{18} \sum_{c=1}^{18} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

Every class contributes exactly $\frac{1}{18} \approx 5.56\%$ to the leaderboard score. **Sinkhorn Balancing** (optimal transport) is applied to match test set predictions to the uniform prior.

---

## 3. Submissions & Iteration Lineage

The Kaggle submission history currently records twelve scored runs:

| # | Version Folder | Description & Key Innovations | Public LB Score |
| :- | :--- | :--- | :---: |
| **V1** | [kernel_v1/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v1) | Baseline: Two-Stage LoRAs + stylometric LinearSVC + centred twins | **0.621** |
| **V2** | [kernel_v2/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v2) | Curated clean passages, punctuation view, and dual-shift ensemble | **0.623** |
| **V3** | [kernel_v3/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v3) | Neural-forward weights and three-point shift blend | **0.616** |
| **V4** | [kernel_v4/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v4) | Uncertainty-gated typographic priors | **0.615** |
| **V5** | [kernel_v5/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v5) | Multi-scale centring, typographic priors, and conservative shift blend | **0.615** |
| **V6** | [kernel_v6/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v6) | Public two-stage ensemble + complementary stack + ATLAS | **0.636** |
| **V7** | [kernel_v7/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v7) | V5 with independent tight/broad embedding centring | **0.618** |
| **V8** | [kernel_v8/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v8) | V2 with only the grouped SVC-shift blend made more conservative | **0.623** |
| **V9** | [kernel_v9/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v9) | V6 complementary stack + ATLAS with V2's curated/punctuation SVC recipe | Submission `57597207` scored **0.635** |
| **V10** | [kernel_v10/](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v10) | V6 + official-only prompt-residual readout with grouped OOF diagnostic | **0.639** (`57603012`) |
| **V11** | [kernel_v11/](./kernel_v11/) | V10 with only the official-only readout weight increased from 5% to 10% | **0.639** (`57621098`) |
| **V12** | [kernel_v12/](./kernel_v12/) | V10 with a three-seed projection ensemble for its 5% official-only readout | **0.639** (`57625772`) |
| **V13** | [kernel_v13/](./kernel_v13/) | V10 with the 1,516-passage GLM/Kimi SVC augmentation | Submission `57637841` pending |
| **V14** | [kernel_v14/](./kernel_v14/) | V10 with the curated public-only SVC pool replacing its original pool | Submission `57637845` pending |
| **V15** | [kernel_v15/](./kernel_v15/) | V14's pool plus GLM/Kimi passages | Submission `57637848` pending |

---

## 4. Results and next experiment

Kaggle's recorded public scores show V10 is currently the best at **0.639**, followed by V6 at **0.636** and V9 at **0.635**. V10 improved on V6 by 0.003, though the public score alone cannot isolate the auxiliary readout's contribution.

One implementation mismatch was found in V5: its multi-scale centring subtracts the broad-cluster mean first, then calculates the tight-cluster mean from the already-adjusted embeddings. The effective correction therefore differs from the documented independent 70% tight / 30% broad calculation. This is a concrete issue to investigate, but the leaderboard alone cannot establish how much it affected the score.

V6 is an exact local reproduction of the public complementary-stack + ATLAS notebook. Its Kaggle competition submission `57591353` completed at **0.636**. The separate three-row smoke output verifies execution and packaging only; it is not the source of that score. See [V6 candidate notes](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v6/README.md).

V7 isolates the V5 multi-scale centring implementation mismatch: both cluster means now come from the original embeddings. Its public score was 0.618, below V2. See [V7 candidate notes](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v7/README.md).

V8 keeps the best-scoring V2 architecture and changes only the final grouped-shift strength mixture. Its three-row fallback output is byte-identical to V2's and does not execute the changed full-test calibration. Competition submission `57593259` scored **0.623**, tying V2 rather than improving it. See [V8 candidate notes](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v8/README.md).

V9 is a V6-derived candidate that preserves its complementary stack and ATLAS path while adding V2's curated SVC passages, punctuation view, and 1.25 temperature. Kaggle GPU smoke execution verified all required complementary members and ATLAS, but used the three-row public preview and did not validate accuracy. V9 was submitted from notebook version 2 as competition submission `57597207` and scored **0.635**, slightly below V6's 0.636. See [V9 candidate notes](file:///c:/Users/ssanj/OneDrive/Desktop/Sanjay/Antigravity/name_that_llm/kernel_v9/README.md).

### V10 and V11

Keep V6's complete recipe as the control: its public source reports **0.642** prompt-grouped CV for the main two-stage blend versus **0.620** on the public leaderboard, while our reproduced submission reached **0.636** after adding the complementary stack and ATLAS. That CV/LB gap warns against fine-grained weight fitting on the small official training set. V9's submission `57597207` scored 0.635, so its SVC-data/features change did not improve on V6. V10 added a 5% official-only prompt-residual readout on the existing LoRA24 feature representation. Its five-fold `GroupKFold` diagnostic reported 0.775411 readout-only OOF balanced accuracy; this neither measures the full ensemble's OOF performance nor validates the leaderboard gain. Full competition submission `57603012` scored **0.639**.

V11 is a one-variable weight ablation: raise the V10 readout contribution from 5% to 10%, holding all other models, data, and settings fixed. The public 0.639 style-readout notebook is explicitly unscored for its new experiment; its attached pretrained readout bank also has an unknown license. V11 therefore uses the V10 official-only readout and does not attach those public weights. Kaggle notebook version 1 completed the three-row public-preview smoke run, with all required model components verified; that is not accuracy evidence. Competition submission `57621098` scored **0.639**. See [V11 candidate notes](./kernel_v11/README.md).

V12 is an independent one-variable branch from V10: keep the 5% mixture weight and all ensemble members fixed, but average three independently seeded 512-dimensional sparse projections in the official-only prompt-residual readout. The Kaggle smoke run reported 0.800616 five-fold prompt-grouped OOF balanced accuracy for the auxiliary ensemble, compared with 0.775411 for V10's single-projection readout. This is not the full ensemble's OOF score and does not guarantee a leaderboard gain. V12 does not use the public experiment's opaque synthetic readout weights. Competition submission `57625772` scored **0.639**. See [V12 candidate notes](./kernel_v12/README.md).

### V13–V15: testing public SVC-pool changes

V13 transfers a public GLM/Kimi SVC-pool experiment that scored 0.640 against a 0.639 control. This transfer is not expected to reproduce that gain automatically because its starting ensemble differs. V14 replaces V10's larger SVC pool with a curated, per-row-provenance public pool; its published five-fold prompt-grouped result is for the SVC component only (0.4582 versus 0.4430), not the full ensemble. V15 tests the curated pool plus the GLM/Kimi rows; their exact-text overlap was checked to be zero, but the combined effect is unknown. All three candidates change only the SVC training pool. Their completed three-row Kaggle previews verified execution and output format only; these previews were not submitted as prediction files. Instead, Kaggle code-competition submissions `57637841` (V13), `57637845` (V14), and `57637848` (V15) were made from kernel version 1 for hidden-test reruns and are pending scores.
