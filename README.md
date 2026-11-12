# Name That LLM — Kaggle Competition

Competition rules, submission workflow, and project process: [competition_info.md](./competition_info.md).
Today's score analysis and submission plan: [action_plan.md](./action_plan.md).

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

The Kaggle submission history records seventeen scored runs through October 26, 2026:

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
| **V13** | [kernel_v13/](./kernel_v13/) | V10 with the 1,516-passage GLM/Kimi SVC augmentation | **0.640** (`57637841`) |
| **V14** | [kernel_v14/](./kernel_v14/) | V10 with the curated public-only SVC pool replacing its original pool | **0.630** (`57637845`) |
| **V15** | [kernel_v15/](./kernel_v15/) | V14's pool plus GLM/Kimi passages | **0.629** (`57637848`) |
| **V17** | [kernel_v17/](./kernel_v17/) | V10 plus GLM-only SVC pool augmentation | **0.640** (`57665156`) |
| **V18** | [kernel_v18/](./kernel_v18/) | V10 plus Kimi-only SVC pool augmentation | **0.639** (`57665192`) |
| **V19** | [kernel_v19/](./kernel_v19/) | Avazaku NLM V27 reproduction | Blocked: required private unattached asset cache (`avazaku/name-that-llm-verified-llama-asset-cache`). |
| **V20** | [kernel_v20/](./kernel_v20/) | NLM V27 + GLM/Kimi SVC augmentation | Blocked: same private asset cache dependency as V19. |
| **V21** | [kernel_v21/](./kernel_v21/) | Pure-pipeline NLM V27 fix | Blocked at runtime due to private LLaMA-3.1-8B shard path assertion. |
| **V22** | [kernel_v22/](./kernel_v22/) | SOTA Sharper Blend (LB 0.643 recipe, Variant 4) | **0.643** (`57830437`) | Clean public source stack. |
| **V23** | [kernel_v23/](./kernel_v23/) | ATLAS 20% + Sharper Blend SOTA (Variant 6) | **0.640** (`57831323`) | Qwen/Helper 15% each with ATLAS 20%. |
| **V24** | [kernel_v24/](./kernel_v24/) | Qwen20 Strong Fulltext Focus (Variant 2) | **0.640** (`57831715`) | Qwen 20%, Helper 10%, ATLAS 10%. |
| **V25** | [kernel_v25/](./kernel_v25/) | Clean 20% ATLAS Expansion (Variant 5) | **0.641** (`57831830`) | Qwen/Helper 10% each, ATLAS 20%. |
| **V26** | [kernel_v26/](./kernel_v26/) | Apex Balanced Trio Blend (Variant 7) | **0.644** (`57831956`) | Qwen 18%, Helper 12%, ATLAS 18%; current best confirmed score. |
| **V27** | [kernel_v27/](./kernel_v27/) | Qwen20 / ATLAS18 blend (Variant 8) | **Pending** (`57874985`) | One blend-weight ablation from V26; full score pending. |
| **V28** | [kernel_v28/](./kernel_v28/) | Qwen16 / Helper12 / ATLAS18 | Preview failed: competition test input missing. |
| **V29** | [kernel_v29/](./kernel_v29/) | Qwen18 / Helper10 / ATLAS18 | **Pending** (`57875299`) | One blend-weight ablation from V26. |
| **V30** | [kernel_v30/](./kernel_v30/) | Qwen18 / Helper12 / ATLAS20 | Preview failed: competition test input missing. |
| **V31** | [kernel_v31/](./kernel_v31/) | Qwen18 / Helper12 / ATLAS16 | Preview failed: competition test input missing. |
| **V38** | [kernel_v38/](./kernel_v38/) | Verbatim alexchilton 0.683 (V28 Sovereign Titan Apex Fusion) | Slate 2026-11-11 |
| **V39** | [kernel_v39/](./kernel_v39/) | V38 with `gsq_share` 0.06 -> 0.12 | Slate 2026-11-11 |
| **V40** | [kernel_v40/](./kernel_v40/) | V38 with WeMM expert share 0.07 -> 0.14 | Slate 2026-11-11 |
| **V41** | [kernel_v41/](./kernel_v41/) | V38 with sinkhorn strength .30 -> .45 | Slate 2026-11-11 |
| **V42** | [kernel_v42/](./kernel_v42/) | V38 with GSQ .12 + WeMM .14 | Slate 2026-11-11 |

V32-V37 (public-recipe reproductions from 2026-11-02) are recorded in
[action_plan.md](./action_plan.md). As of 2026-11-11 the public-notebook ceiling is 0.684 while the
leaderboard top is 0.765; see the 2026-11-11 entry in the action plan for why that gap is a data
gap, not a blend-weight gap.

---

## 4. Results and next experiment

Kaggle's latest recorded public scores show V26 is currently best at **0.644** (`57831956`), ahead of V22 at 0.643, V25 at 0.641, and V23/V24 at 0.640. Earlier V13/V17 scored 0.640. These are small public-score differences; no paired grouped-OOF or per-class evidence currently establishes that V26's blend is robustly better.

V17 and V18 are single-source SVC-pool ablations from V10: V17 adds 1,200 GLM passages and V18 adds 316 Kimi passages. Their code-competition submissions completed on October 26, 2026: V17 (`57665156`) scored 0.640 and V18 (`57665192`) scored 0.639. V13 remains tied for the best confirmed score at 0.640. See [V17 notes](./kernel_v17/README.md) and [V18 notes](./kernel_v18/README.md).

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

V13 transfers a public GLM/Kimi SVC-pool experiment and scored 0.640 against V10's 0.639. V14 replaces V10's larger SVC pool with a curated, per-row-provenance public pool; its published five-fold prompt-grouped result was for the SVC component only (0.4582 versus 0.4430), not the full ensemble. V14 scored 0.630. V15 tests the curated pool plus the GLM/Kimi rows; exact-text overlap between the added corpora was checked to be zero, but the combination scored 0.629. Kaggle submissions `57637841` (V13), `57637845` (V14), and `57637848` (V15) are complete and scored. Their visible three-row previews verified execution and output formatting only; they were not the scored predictions. See [today's action plan](./action_plan.md) for controlled follow-up experiments.

### V16 grouped SVC source ablation — completed October 26, 2026

The Kaggle validation completed after 10,378 seconds and wrote `kernel_v16/output/v16_svc_source_ablation.json`. Five prompt-grouped folds produced SVC-only OOF balanced accuracy of 0.473400 for the base pool, 0.473400 for base+GLM, 0.473773 for base+Kimi, and **0.476734 for base+GLM+Kimi** (+0.003333 versus base). This is a component metric, not full-ensemble OOF or a leaderboard score. V16 produces no competition submission file and must not be submitted as an inference notebook. V13 and V17 scored 0.640; V18 scored 0.639. Keep 0.640 as the incumbent until a complete full-pipeline validation supports a new candidate.

## V19/V20 plan and Kaggle run status — October 26, 2026

The latest public lead is the [verified NLM V27 reproduction](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction), whose author documents a verified 0.651 competition score; it is not our score. V19 reproduces that notebook with unchanged executable code and pinned Kaggle inputs. V20 changes only the SVC pool by adding the 1,516 CC BY 4.0 GLM/Kimi passages after exact `(model, text)` deduplication. The V20 idea is supported only by SVC-component OOF and V13's modest full-pipeline gain, so treat it as an experiment. The previous full runs were cancelled due to long runtimes; their smoke runs are setup checks only and cannot be submitted for scoring. New full Kaggle runs for V19 and V20 are now running as kernel version 4; submit only after their outputs and completion are verified.

Both notebooks run on Kaggle GPU; no local model training is performed. The public reproduction reports a 4h04 visible run on T4 x2 and an unknown hidden-run time. As of October 26, 2026, Kaggle reports 2 submissions used today and 3 remaining. The version 4 V19 and V20 runs are intended for next-day review; preserve the quota until either run completes and its output is verified.

---

## 2026-11-12 - the 0.684 pipeline slate

The previous slate (v38-v42) forked the **0.683** Apex-Sovereign notebook; it peaked at
0.678 and the knob ladder inside that lineage is exhausted. The public **0.684** release
is a different pipeline: `infer.py` + `comp_stack.py`, the native Qwen3-Embedding-8B
expert at share 0.40, the Qwen3.8-27B classifier at 3%, a run-time-fitted complementary
stack (`comp_w 0.15`), embedding-similarity passage grouping, and no WeMM expert and no
hard typography rules. Two independent public copies score 0.684, and this account had
never run it.

| v | kernel | setting vs 0.684 |
|---|---|---|
| v43 | `ntl-v43-684-verbatim` | none (canonical reproduction draw) |
| v44 | `ntl-v44-684-sinkhorn20` | Sinkhorn strength 0.30 -> 0.20 |
| v45 | `ntl-v45-684-targetmix50` | Sinkhorn target mix 0.35 -> 0.50 |
| v46 | `ntl-v46-684-punct` | training-answer punctuation prior (new mechanism) |
| v47 | `ntl-v47-684-native45` | native expert share 0.40 -> 0.45 |

Built by `tools_build_684.py` from the public notebooks' own cells with the authors'
attachment lists (read from the pull API by `tools/fetch_public_meta.py`); driven by
`tools/run_slate_684.py`, which pushes one kernel at a time, insists on a 3-row
`submission.parquet` with `[id, model]` before spending a slot, and submits through
`competition_submit_code` (the CLI's `submit -k` path returns an empty-body 400 here).

### Own LoRA member

The Qwen3-8B QLoRA trainer (`kernel_t1_q8_lora`, driven by `trainer/train_q8_lora.py`)
failed on its first run with `FileNotFoundError: /kaggle/input/ntl-pool-v4/pool_v4.parquet`
- the dataset mounted one level deeper than expected. `find_data()` now tries both known
layouts and then a bounded recursive glob, so a mount-layout change cannot kill the run
again. The trainer is unchanged otherwise: class-balanced sampling, prompt-grouped
holdouts on both corpora, per-class recall and top-3 confusions written out.
