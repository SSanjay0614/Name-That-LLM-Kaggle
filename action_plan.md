# Name That LLM — Action Plan for October 26, 2026

This is an experiment and submission plan only. No model or kernel code has been changed for this plan.

## Current facts

Kaggle CLI was checked on October 26, 2026. It reported **5 submissions remaining today** and **15 lifetime submissions**. Recheck the quota immediately before any submission.

The three latest scored submissions were submitted on October 25 UTC:

| Candidate | Submission | Public score | Change from V10 |
|---|---:|---:|---:|
| V13 — V10 plus 1,516 GLM/Kimi SVC passages | `57637841` | **0.640** | **+0.001** |
| V14 — V10 with curated public-only SVC pool replacing the original pool | `57637845` | **0.630** | **-0.009** |
| V15 — curated pool plus GLM/Kimi passages | `57637848` | **0.629** | **-0.010** |

The previous best, V10/V11/V12, scored **0.639**. V6 scored **0.636**. V13 is the current best confirmed result, but its 0.001 lead over V10 is small and may reflect leaderboard sampling noise. No class-level score breakdown is available here, so the public scores do not identify which classes or passages caused the changes.

## What yesterday's tests tell us

### V13: keep the GLM/Kimi augmentation as the current anchor

V13 changed only the SVC training pool relative to V10 by adding 1,516 passages from GLM-5.3, GLM-5.3-flash, and Kimi-K3. It scored 0.640, one point in the third decimal above V10's 0.639. This is a positive but modest result—not evidence that the change will generalize reliably or that every added model's passages helped.

### V14: do not repeat the curated-pool replacement

V14 replaced the 75,783-row V10 SVC pool with a 40,129-row curated public pool and fell to 0.630. The curated pool's reported prompt-grouped OOF gain (0.4582 vs. 0.4430) applied to the **SVC component only**, not the complete neural/SVC/ATLAS ensemble. A component can score better in isolation yet contribute less useful diversity or coverage when mixed into the full system. The result is a warning against treating component OOF as a proxy for end-to-end leaderboard performance.

The score alone does not prove why V14 fell. Plausible hypotheses to test—not established causes—include reduced coverage after shrinking the pool, differences between the curated validation distribution and hidden test distribution, or less complementary errors relative to the unchanged neural ensemble.

### V15: the GLM/Kimi rows did not rescue the curated pool

V15 combined the curated pool with the GLM/Kimi passages. It scored 0.629, one point below V14 and 0.010 below V13. The exact-text overlap check between the two added corpora being zero only establishes that their passages differ; it does not establish that their errors are complementary or that the replacement base pool is sound.

### Main takeaway

The strongest evidence from these three runs favors **adding the GLM/Kimi data to the existing V10 pool**, not replacing that pool with the curated source. Keep the V13 lineage as the incumbent. Do not spend an attempt on another V14/V15-style full-pool replacement without new full-pipeline validation.

## Plan for today's attempts

### Stage 1 — use the existing evidence before spending submissions

1. Treat V13 (0.640) as the current control and preserve V10 (0.639) as the paired baseline.
2. Compare V10 and V13 with the **same prompt-grouped out-of-fold folds** using the complete ensemble, not the SVC branch alone. Save overall balanced accuracy, per-class recall, confusion matrices, and the V13-minus-V10 delta for each fold.
3. Run two data-isolation variants against those exact folds:
   - V10 pool + GLM passages only (exclude Kimi).
   - V10 pool + Kimi passages only (exclude GLM).
4. Check passage provenance, duplicate/near-duplicate contamination, per-class counts, and whether the added rows disproportionately alter the two GLM labels or the Kimi label. Do not infer this from aggregate leaderboard scores.
5. If the full-pipeline grouped OOF differences are unstable across folds or do not identify a clear contributor, **do not submit a speculative variant**. Keeping the 0.640 incumbent is preferable to using all five daily attempts.

### Stage 2 — conditional candidate order

Only proceed to competition submissions after the complete-pipeline validation and Kaggle smoke run both pass.

1. **First candidate:** V13's original pool plus only the GLM or Kimi subset that shows the more consistent full-pipeline grouped-OOF gain. Keep every other model, blend weight, and calibration step unchanged. This isolates which augmentation produced V13's small observed gain.
2. **Second candidate, only if supported:** the other single-source ablation, or a single SVC-mixture-weight adjustment chosen from grouped OOF. Do not combine a new corpus, new blend weight, and calibration change in one candidate.
3. **Do not submit:** V14's curated-pool replacement or V15's curated-plus-GLM/Kimi recipe as-is; both are already lower-scoring and provide no evidence that resubmitting them will improve today's result.
4. **Stop after at most two new competition submissions today** unless a candidate shows clear, reproducible full-pipeline validation improvement and there is a specific reason to use another attempt. Five available attempts are a ceiling, not a target.

If today's OOF and end-to-end Kaggle run cannot be completed, make no new submission and keep V13 as the best known candidate. A repeated V13 submission is useful only if needed as an operational/final-selection choice—not as an accuracy experiment.

## Required validation gate for each candidate

- The candidate differs from its parent in one documented variable.
- Grouped OOF compares the **full ensemble** against V10 and V13 on identical folds; report per-class recall as well as the aggregate.
- All added passages have recorded source and license provenance; no hidden test labels or leaderboard-derived labels are used.
- Kaggle smoke execution loads every dependency and processes the complete competition test input for the actual code rerun.
- The output has every expected row and legal labels in the required order/schema; a small visible preview is only a packaging check.
- The completed kernel version and output artifact are confirmed before using a submission attempt.
- Record submission ID, score, and any failures in the README and candidate notes. Do not call an OOF result or smoke run a leaderboard score.

## Recommendation at the start of October 26 (historical snapshot)

**Anchor on V13. First investigate GLM-only versus Kimi-only augmentation with identical prompt-grouped full-ensemble OOF.** If one source gives a stable gain, submit that single-source ablation first, then consider one further isolated candidate. Avoid the curated replacement path and preserve at least three of today's five attempts unless evidence justifies more.

## October 26 live-check update

- The initial live check showed **0 submissions today, 15 lifetime, and 5 remaining today**. V13 (`57637841`) remained the best confirmed score at 0.640; V14/V15 remained 0.630/0.629.
- Kaggle's public kernel list showed Areeb Ali's 0.643 reweighting notebook. Its reported change is a 15% Qwen3.8-27B share and 15% spectral-head share on a different 0.642 stack. The notebook's credits say the attached spectral-head dataset has an unspecified license. We did not copy its code or assets; it is a research reference only under this project's provenance rule.
- V16 is a Kaggle validation notebook for five-fold `prompt_id` grouped **SVC-component-only** comparison of the original pool, GLM-only, Kimi-only, and GLM+Kimi. Its first run stopped before fitting because `pool_v4.parquet` was not attached. Version 3 now attaches the verified `nishantdahal/ntl-handwriting-blend` source containing that file and the CC BY 4.0 GLM/Kimi dataset. It remains `RUNNING` with no log output or visible fold marker. It is CPU-only and fits four large TF-IDF views for each of 4 recipes across 5 folds: 80 high-dimensional vectorizer fits over the 75,783-row pool plus official training rows. That workload explains a multi-hour runtime; the blank logs leave it unclear whether Kaggle is still preparing inputs or has reached fitting. Its output is diagnostic only and cannot pass the full-ensemble OOF submission gate.
- V16's baseline pool carries CC BY-SA 4.0 terms; its README records that alongside the added dataset's CC BY 4.0 attribution. No local model training was run.
- V17 and V18 are the requested one-source V10 descendants: GLM-only (1,200 rows) and Kimi-only (316 rows). Their cloud smoke kernels verified the full set of ensemble components on the three-row preview. Their later code submissions completed at 0.640 and 0.639, respectively; see the October 26 follow-up below.

Next: retrieve V17/V18 scores when Kaggle completes scoring. Retrieve V16 output when its grouped component ablation finishes; treat those SVC results as source-screening only. Public leaderboard differences should be treated as noisy evidence, not full-ensemble OOF.

## Follow-up plan — October 26, 2026

At the October 26 checkpoint, the live quota showed **3 submissions remaining** (2 used). To match the requested two-run budget, reserve one attempt and prepare:

1. **V19 — verified NLM V27 reproduction.** The public Avazaku reproduction documents an official **0.651** score, verified for its exact V3/version 362281064 and matching Dhruv Tibarewal's scored V27 V1. Our fork keeps the inference code and pinned inputs unchanged; the result is not ours until it completes a hidden-test submission.
2. **V20 — V27 plus GLM/Kimi SVC pool rows.** Change only the SVC pool by appending the 1,516 CC BY 4.0 passages, deduplicated against the public pool by exact `(model, text)`. This is motivated by V16's SVC-only +0.00333 balanced-accuracy delta for both sources and V13's full-pipeline 0.640 versus V10's 0.639; neither result guarantees transfer to V27.

The V27 reproduction's notebook reports a 4h04 visible run on Kaggle T4 x2, under the current nine-hour GPU limit, but reports no hidden-run duration. Run both variants only in Kaggle, preserve input licenses and attribution, and verify full hidden-test output before spending submissions. The first upload attempt exposed a renderer validation problem caused by added cell IDs; the corrected saved versions omit IDs from the added cells. Do not treat the failed initial version as a completed run.

## References

- [Competition workflow and rules](./competition_info.md)
- [Submission history and experiment lineage](./README.md)
- [V10 baseline notes](./kernel_v10/README.md)
- [V13 GLM/Kimi augmentation notes](./kernel_v13/README.md)
- [V14 curated-pool notes](./kernel_v14/README.md)
- [V15 combined-pool notes](./kernel_v15/README.md)

## V16 completion — October 26, 2026

Kaggle kernel `sanjay0614/ntl-v16-grouped-svc-source-ablation` completed after 10,378 seconds. It is a five-fold `prompt_id` grouped **SVC-component-only** diagnostic and wrote `v16_svc_source_ablation.json`; it does not create a competition prediction file, so there is no valid V16 artifact to submit.

| SVC training pool | Grouped OOF balanced accuracy | Delta vs base |
|---|---:|---:|
| Base public pool | 0.473400 | — |
| Base + GLM passages | 0.473400 | +0.000000 |
| Base + Kimi passages | 0.473773 | +0.000373 |
| Base + GLM + Kimi | **0.476734** | **+0.003333** |

The both-source pool is the best SVC-only result, but it is not full-ensemble OOF and cannot be converted into a leaderboard score. The existing combined-source full pipeline V13 scored **0.640**; the later V17 GLM-only and V18 Kimi-only submissions scored **0.640** and **0.639** (refs `57665156` and `57665192`). Keep 0.640 as the incumbent until a new complete full-pipeline validation is run. At this October 26 checkpoint, NTL quota was **3 submissions remaining**.

## Yesterday's snapshot - October 26, 2026 (historical)

- Live Kaggle CLI confirms V17 (`57665156`) completed at **0.640** and V18 (`57665192`) completed at **0.639**. V13 and V17 tie for the best confirmed score at 0.640.
- The Name That LLM quota at this October 26 snapshot was **3 submissions remaining** (2 used).
- The previous V19 and V20 full inference runs were cancelled due to long runtime. Their smoke notebooks are not scoreable submissions. V20 kernel version 3 had no saved `submission.parquet`, so the CLI correctly rejected that version.
- On October 26, full GPU Kaggle runs for V19 and V20 were pushed as kernel version 4. Both are confirmed `RUNNING`. Review logs and outputs tomorrow; submit only after the correct version completes and its output is verified. The full V27 recipe previously took over four hours, so do not expect a quick result.
- No quick, validated, unsubmitted improvement was available today. The 3 remaining attempts are reserved until V19/V20 finish and their outputs can be assessed.

## Yesterday's V19/V20 run - launched October 26, reviewed October 27

- V19 and V20 kernel version 4 were launched on October 26. By October 27, each had run for **more than 9 hours** (user-observed) and consumed substantial Kaggle GPU time. Kaggle now reports both as `CANCEL_ACKNOWLEDGED`; neither produced a verified submission output, and neither was submitted.
- Treat the full V27 recipe in its current form as over the practical runtime budget. Do not queue these full versions again unchanged. Any revisit needs profiling and a concrete runtime reduction plan first, with progress checkpoints and an early stop threshold before the 9-hour cap.
- No competition attempts were spent on V19/V20. On October 27, Kaggle reports **5 submissions available, 0 used today** for Name That LLM.

## Today's action plan - October 27, 2026 (Name That LLM only)

### Current state

- Best confirmed score: **0.640**, tied by V13 and V17. V18 scored 0.639.
- Kaggle quota: **5 available, 0 used today**. V19/V20 version 4 are cancelled and have no verified submission outputs.
- The public verified NLM V27 reproduction reports 0.651, but our V19/V20 versions each exceeded 9 hours. Do not relaunch that recipe unchanged.
- A newer public notebook, [0.642 | LoRA + Llama + Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b), is Apache 2.0 and reports 0.642. Its visible 23-second run is not evidence of a fast full-test run: the notebook explicitly shortcuts only the exact three-row public fixture, while the hidden evaluation takes the full ensemble path. Its attached spectral-head dataset has unresolved license terms; treat that component as research-only until audited. The separate 0.643 sharper-blend notebook also uses spectral-head assets with unspecified licensing and must not be copied as-is.

### Goal for today

Find one evidence-backed change that can beat 0.640 and finish within Kaggle's 9-hour notebook limit. Preserve the remaining attempts if no candidate clears the validation and runtime gates.

### Work order

1. **Diagnose the V19/V20 runtime before launching another full GPU job.** Review their final logs and notebook stages to identify where the time accumulated (input preparation, model loading, feature extraction, or inference). Record stage timings and any progress gaps. Use Kaggle Cloud for any timing probe; do not run model processing locally.
2. **Audit the 0.642 public notebook as a source of ideas.** Compare its model contributions and blend weights against V13/V17. Verify the exact hidden-run execution path and the licenses for every attached dataset/model. Do not use the spectral-head asset while its license is unresolved; prefer a small, independently implementable change using already approved inputs.
3. **Build at most one isolated candidate from the 0.640 V13/V17 control.** Keep the full ensemble fixed except for one justified change. Require prompt-grouped, full-pipeline OOF against the same folds before treating it as a scoring candidate; the V16 SVC-only ablation is insufficient evidence by itself.
4. **Run a bounded Kaggle Cloud validation only after stages 1-3 pass.** Use the configured offline T4 x2 environment, add visible progress checkpoints, and stop if the projected total exceeds 8 hours or a long stage stops advancing. Require the full hidden-test path and valid complete `submission.parquet` before submitting.
5. **Submission budget:** make at most one new NTL submission today, and only after the candidate clears both validation and runtime gates. Recheck quota immediately before submission; do not use attempts on a smoke run, a duplicate, or an unverified public-score claim.

If no candidate clears these gates today, spend the day on runtime diagnosis and a license-cleared candidate design, then keep the 0.640 result as the incumbent.

---

## STOP NOTICE - all three competitions paused until 2026-10-31 00:00 UTC (added October 27, 2026)

All three active competitions are at a stop, and the reason is the same for each: the Kaggle GPU
quota is exhausted.

- GPU hours used: 40.52 h out of 30 h. Remaining: **0.00 h**. The allowance refreshes on **2026-10-31 00:00 UTC**.
- Every submission path in this project needs a GPU. Kaggle re-runs a notebook-backed submission on the
  hidden data, so nothing can be submitted from local CPU work or from an already cached output.
- Therefore no submission can be made to any of the three competitions until the reset.

What each competition is waiting for:

- **ARC-AGI-3** (closes 2026-11-02 23:59 UTC): gets first call on the refreshed quota. Three scored
  attempts on Oct 31, Nov 1 and Nov 2, using the Franzen setup that already passed CPU validation. This
  competition has the earliest deadline, so it takes priority over the other two.
- **Enveda CASMI26** (closes 2026-12-14): resumes Nov 3, leaving about six weeks at roughly 30 h per week.
  Incumbent score stays at 0.470.
- **Name That LLM** (closes 2027-01-14): resumes after ARC. Incumbent score stays at 0.640. A full-recipe
  CPU run does not fit the 9 h limit and would score near 0.48 anyway, so it is not an option.

Nothing above retracts earlier content in this file. It only records the current pause.
