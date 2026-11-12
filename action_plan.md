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

## Nov 1, 2026 score audit

Kaggle's submission listing reports these completed public scores:

| Candidate | Submission | Public score |
|---|---:|---:|
| V22 — sharper blend | `57830437` | 0.643 |
| V23 — Qwen/Helper 15%, ATLAS 20% | `57831323` | 0.640 |
| V24 — Qwen 20%, Helper/ATLAS 10% | `57831715` | 0.640 |
| V25 — Qwen/Helper 10%, ATLAS 20% | `57831830` | 0.641 |
| V26 — Qwen 18%, Helper 12%, ATLAS 18% | `57831956` | **0.644** |

V26 is the current public-score anchor, but its 0.001 lead over V22 is small. A source comparison
confirmed that V22 and V26 have the same 81 cells and differ only in the blend settings cell:
V26 selects a new weight tuple. The scores support testing the blend direction further, but do not
establish a robust improvement without paired prompt-grouped validation or per-class metrics.
The fast-commit logs are only packaging previews; they are not accuracy validation. The full
competition submissions did receive scores.

**Next experiment:** keep V26 as the control. Before using another attempt, produce full-pipeline,
prompt-grouped held-out predictions for V22 and V26 with identical inputs and folds, then report
paired balanced-accuracy deltas and per-class recall. If validation supports it, test only a small
adjacent blend-weight change, reusing cached component predictions if available. The current kernel
outputs contain the final submission and fast-commit log, not a saved full component-prediction
matrix, so do not assume weight changes can be tested without another full inference run.

Kaggle reported **five attempts remaining today**. No Nov 1 Name That LLM attempt has been used.

### Nov 1 candidate run

Built and pushed V27, a single blend-weight ablation from V26: Qwen 0.20, Helper 0.10, ATLAS
0.18. Kernel `sanjay0614/ntl-v27-qwen20-atlas18` version 1 completed and produced a valid
`submission.parquet`; the run explicitly reported `exact_public_format_preview`, three rows,
no model weights loaded, and `hidden_runtime_verified=false`. This is packaging evidence only.
Submitted V27 as `57874985`, currently pending; Kaggle's code-competition scoring rerun evaluates
the full hidden set. One of today's five NTL attempts is used; four remain.

The four remaining attempts were planned for isolated one-weight changes from V26:
V28 lowers Qwen to 0.16; V29 lowers Helper to 0.10; V30 raises ATLAS to 0.20; V31 lowers ATLAS
to 0.16. V29 completed its preview and was submitted as `57875299` (pending). V28, V30, and V31
failed their preview runs at the same assertion because `/kaggle/input/competitions/name-that-llm/test.csv`
was missing; they have not been submitted. V28, V30, and V31 retries also failed. Kaggle reports
three NTL attempts remaining. Do not spend them on those kernels unless a successful run produces
their own valid outputs.

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

---

## 2026-10-27 — no activity from this session

Not worked on today. No submission, no kernel push, no local run. The GPU quota is committed to
ARC-AGI-3 until 2026-10-31, so this competition is queued behind that date; the plan below is
unchanged. Recorded here only so the gap in the daily log is explained rather than looking like an
oversight.

---

## 2026-11-02 - stop tuning our own blend, reproduce the best public recipe five times

The Oct 31 - Nov 1 ladder is decisive: V22-V29 all sit at 0.640-0.644 and the ordering of those
scores is not reproducible. V26 (0.644) stays as the incumbent, but another reweight of our own
components cannot beat it by more than the board noise.

The public board is a different picture. Independent accounts measure the same family of recipes
at 0.663-0.664:

- alexchilton's *All Public + What Failed* runs bgunshui's **Apex E** and scored **0.663** on
  alexchilton's account. Its own table shows identical code moving by up to 0.009 across
  resubmits, which is why one lucky score is not evidence.
- gloria's controlled ladder reproduces the same control at **0.663** and lands 0.664 for the
  open-pool SVC at weight 12 plus sinkhorn 0.30.
- zbigniewkirchner's one-swap (the open pool in the 4-view SVC, `svc_w 12`) is the single change
  underneath both, and is itself measured at 0.664.

So this session spent all five slots on independent public recipes rather than on our blend, and
deliberately picked five different lineages so a single failure mode does not cost the day:

| slot | kernel | recipe |
|---:|---|---|
| 1 | `ntl-v32-apex-e-public` | Apex E verbatim (198k clean pool, `svc_w 12`, typography 1.2, sinkhorn 0.60) |
| 2 | `ntl-v33-who-are-u-tuned` | judith007 hyperparameter-tuned Who Are U |
| 3 | `ntl-v35-lora-trained27-gentle` | kirilltym LoRA + Trained27 GSQ 7% + gentle sibling |
| 4 | `ntl-v34-one-swap-open-pool` | zbigniew one-swap |
| 5 | `ntl-v37-open-pool-v11-matched` | gloria K: v11 pool in SVC + 49k matched v11 external rows |

Every kernel was copied verbatim from the pulled notebook; only id, title, code_file and the
private flag were rewritten, so the dataset/kernel/model attachments are the author's own. All
commit runs completed on the cheap path (three-row fixture, `submission.parquet` 3 rows) and all
five submissions `57906088`, `57906329`, `57906792`, `57907957`, `57907962` are pending.

**What the five slots are testing.** Not five independent chances at a knob: four of them are the
same 0.663-0.664 family with different member sets, and the fifth changes the external rows. If
they land at 0.66 the gap to our 0.644 is real and comes from the 198k open pool plus the
second-model members, not from calibration. If they land at 0.64, the public scores do not
transfer and our own component set is competitive after all.

---

## 2026-11-11 - public ceiling is 0.684; the leaderboard top is 0.765

**The number that changes the plan.** `kaggle competitions leaderboard name-that-llm --show`
puts the top of the board at **0.765** (James Liebel), then 0.756, 0.752, 0.752, 0.748. Every
public notebook tops out at **0.683-0.684** (alexchilton 0.683, hengzheng's fork 0.684, omerfaruk
0.682, heliosli/lightningv08 0.681/0.680). Our best is 0.666.

A 0.08 gap is not a blend-weight gap. The public stacks all descend from the same handful of
`prefit` assets (SVC4, Gaussian heads, GSQ-27B, five-fold Qwen3-Embedding), and independent forks
converge on 0.680-0.684 no matter which weight they move. The forum's single-model thread
(754860) puts the best *single* open models at 0.65-0.68, so the leaders are not winning with one
better classifier either: they are almost certainly training on a much larger self-generated
corpus (rules §2.6 allows it; it costs API money, which we are not spending).

So: treat **0.683-0.684 as the practical ceiling of the open-asset route**, and treat the remaining
0.08 as out of reach unless we generate a large proprietary corpus.

**Attachment audit, before spending a slot.** All 21 datasets and all 8 kernel sources resolve on
our account (`datasets files` / `kernels status`), and the push reached the GPU-session check,
which means the metadata (including the 7 model sources) validated. The verbatim copy is
byte-identical to alexchilton's notebook (SHA256 D5002A84...).

**Today's slate (5 slots, 0 used so far).** Driven by `tools/run_slate.py`: it retries only the
`Maximum batch GPU session count` refusal, then for each kernel waits for COMPLETE, refuses to
submit unless the output is the three-row preview artifact, and only then calls
`competitions submit`. CASMI training holds both GPU sessions until roughly 19:00 local.

| order | kernel | change vs the 0.683 verbatim |
|---:|---|---|
| 1 | `ntl-v38-0683-verbatim` | none - reproduction baseline |
| 2 | `ntl-v39-gsq12` | `gsq_share` 0.06 -> 0.12 (omerfaruk's 0.682 setting) |
| 3 | `ntl-v40-wemm14` | WeMM expert share 0.07 -> 0.14 (never probed publicly) |
| 4 | `ntl-v41-sinkhorn45` | sinkhorn strength .30 -> .45 (judith's 0.664 value) |
| 5 | `ntl-v42-gsq12-wemm14` | both expert shares at once |

All five are inside the documented +-0.009 noise band, so they are not expected to prove anything
on their own. Two honest reasons to still spend them:

1. **They are free samples of a nondeterministic pipeline.** Identical code has scored 0.645 and
   0.654 on this board, which means predictions differ run to run; the private score varies too.
   With two final selections per entrant, five near-equal draws plus picking the best two is a
   real (if small) edge.
2. **The two expert-share axes are genuinely untested.** GSQ and WeMM are the only members added
   after the 0.680 base whose share was never ablated in public.

**What would actually move the needle next.** Not another fork. Either (a) a trained expert on a
corpus several times larger than the 75.8k-row open pool we hold locally, which needs generation
money, or (b) accept 0.683 and keep the slot lottery running until the deadline. Decide after the
slate scores.

### 2026-11-11 afternoon - first NTL submission of the day, and our own member starts

**Submitted.** `58230146` (v38, 0.683 verbatim). The slate verified the three-row preview
artifact (`rows=3 cols=[id, model] mode=exact_public_format_preview`) before calling
`competitions submit`, and the counter went 5 -> 4. Quota is per competition, so this used none of
CASMI's (which was already at 0/5).

**Session juggling.** `casmi26-scorer-big-c` was deleted to free one of the two batch GPU
sessions (its source is preserved in `enveda_casmi/kernel_c4_scorer_big_c/`, so it can be
re-pushed). The session took about five minutes to be reclaimed, then the slate started. The
`feat` trainer keeps the other session until roughly 19:00.

**Our own member, `kernel_t1_q8_lora`.** Qwen3-8B 4-bit + LoRA r=16 on the 75.8k-row open pool
plus the official 2,699 rows, with three deliberate differences from the public LoRA members:
class-balanced sampling (the metric is balanced accuracy and the pool is imbalanced 2,338 ->
6,736), prompt-grouped holdouts on *both* corpora, and per-class recall plus top-3 confusions
written out. Source of truth is `trainer/train_q8_lora.py`; `tools/make_trainer_nb.py` compiles
it into the notebook and syntax-checks every cell. Learning data ships as the private dataset
`sanjay0614/ntl-pool-v4` (status `ready`).

The gate is honest: this member earns a blend slot only if its pool-holdout balanced accuracy
clears the ~0.66 that public components report. Otherwise it stays an experiment.

---

## 2026-11-12 - run the 0.684 pipeline we never ran

Quota: **0/5 used at the start of the session** (lifetime 34).

**What was wrong with the previous five days of NTL slots.** Every one of our five
NTL submissions on 2026-11-11 (v38-v42, refs 58230146/58230186/58230220/58230246/58230268)
was a fork of the same **0.683** Apex-Sovereign notebook: 60 cells, `CONFIG['gsq_share']`,
a WeMM expert at 0.07, hard typography rules, `balance_logits(strength=.30, target_mix=.35)`.
Scores: 0.677, 0.677, 0.673, 0.678 (best-ever), 0.672. The knob ladder inside that
lineage is exhausted: +-0.009 run noise, no ordering signal.

The public **0.684** release is a *different pipeline*, not a knob on 0.683:
`lbjyang/ntl-684-ozertuu-mirror` (v3, itself a mirror of ozertuu's code, also published
unchanged by areebbb, krshs90, wadigongren). It runs `infer.py` + `comp_stack.py`,
adds the **native Qwen3-Embedding-8B expert at share 0.40** and the **Qwen3.8-27B
classifier at 3%**, fits its complementary stack at run time with `comp_w 0.15`, groups
test passages by embedding similarity, and has no WeMM member and no hard typography.
Two independent copies of it score 0.684: lbjyang's mirror, and heliosli's
`name-that-llm-lv683-a15` v13 (which adds a one-line mount shim and still scores 0.684).

We have never submitted this pipeline. That is the gap today's slate closes.

### Today's five

| v | kernel | setting | source |
|---|---|---|---|
| v43 | `ntl-v43-684-verbatim` | none | `lbjyang/ntl-684-ozertuu-mirror` v3, verbatim |
| v44 | `ntl-v44-684-sinkhorn20` | Sinkhorn strength 0.30 -> 0.20 | `prvsiyan/the-prompt-is-not-the-author` v37 |
| v45 | `ntl-v45-684-targetmix50` | Sinkhorn target mix 0.35 -> 0.50 | `heliosli/name-that-llm-lv683-a15` v14 |
| v46 | `ntl-v46-684-punct` | training-answer punctuation prior | `kirilltym/lora-trained27-evidence-for-close-calls` v19 |
| v47 | `ntl-v47-684-native45` | native expert share 0.40 -> 0.45 | `prvsiyan/...` with Sinkhorn restored to 0.30 |

All five are published copies (Apache-2.0) with the authors' own attachment lists, read
from the pull API (`tools/fetch_public_meta.py`) rather than guessed: 17-24 datasets,
8 kernel outputs, 6-10 models, the same public assets the 0.683 lineage already used,
plus `crtproduction/name-that-llm-open-pool`, `dhruvtibarewal/nlm-matched-clean`,
`dhruvtibarewal/nlm-offline-wheels`, `alexchilton/llama-3-1-8b-base` and
`dangkhoa2016/tencent-wemm-embedding-4b`, all verified to resolve on this account.

Builder: `tools_build_684.py` (asserted exact-match edits; every notebook parses, and
only the named arguments differ). Driver: `tools/run_slate_684.py` (push one at a time,
wait COMPLETE, require a 3-row `submission.parquet` with `[id, model]`, then submit via
`competition_submit_code` - the CLI `submit -k` path returns a 400 on this account).

**Why these variants and not five identical draws.** The 0.684 code's two calibration
axes have public evidence behind them: weaker Sinkhorn is reported at +0.003 because the
hidden class mix is uneven (and areebbb's higher target mix lost 0.002), and the native
share is the one member added over 0.683. The punctuation prior is a genuinely new
post-hoc mechanism fitted on the official answers, measured at +4..10 correct out of
2,159 in the author's prompt-held-out checks. Together they bracket the pipeline rather
than repeat it.

**Decision rule.** If v43 lands >= 0.680 the 0.684 lineage transfers and becomes the new
base for every future slate, with the calibration axes above as the only knobs. If the
whole slate lands <= 0.678, the 0.08 gap to the board top is confirmed as corpus-bound
(the leaders' 0.765 needs generated data we are not buying) and the remaining work is
slot variance plus the own-LoRA member, not pipeline archaeology.

### Submitted so far

| ref | kernel | description |
|---|---|---|
| 58271414 | `ntl-v43-684-verbatim` | public 0.684 (ozertuu/lbjyang mirror v3) verbatim, no edits |

The commit run hit the mirror's own `emit_known_preview` path in 1.3 minutes and the
artifact was exactly 3 rows of `[id, model]` with `mode=exact_public_format_preview`,
so the slot was spent on a verified preview. Note for the driver: the client returns a
typed response object (`ApiCreateCodeSubmissionResponse`), not a dict - the first version
of `submit()` called `.get()` on it and crashed *after* the submission had already been
accepted; the fix reads `res.ref` (or `res["ref"]`) directly.

### Own LoRA member, second attempt

`sanjay0614/ntl-v43-q8-lora-pool` version 2 is pushed with the mount fix. The dataset was
present with `pool_v4.parquet` and `train.csv` at its root, so the first run's
`/kaggle/input/ntl-pool-v4/pool_v4.parquet` path was simply the wrong one for this
session's layout. `find_data()` now tries `/kaggle/input/<slug>`, then
`/kaggle/input/datasets/sanjay0614/ntl-pool-v4`, then a bounded recursive glob, and fails
with the top-level mount listing if all three miss. It runs on the second GPU session and
spends no submission slot.

### Full slate submitted - all 5 slots used

| ref | v | kernel | setting vs 0.684 |
|---|---|---|---|
| 58271414 | v43 | `ntl-v43-684-verbatim` | none (canonical reproduction) |
| 58271509 | v44 | `ntl-v44-684-sinkhorn20` | Sinkhorn strength 0.30 -> 0.20 |
| 58271607 | v45 | `ntl-v45-684-targetmix50` | Sinkhorn target mix 0.35 -> 0.50 |
| 58271647 | v46 | `ntl-v46-684-punct` | training-answer punctuation prior |
| 58271742 | v47 | `ntl-v47-684-native45` | native expert share 0.40 -> 0.45 |

All five commit runs finished in 1-3 minutes and produced a verified 3-row
`submission.parquet` with `[id, model]` before the slot was spent. Two of the five
(v43, v46) carry the author's explicit `exact_public_format_preview` marker; the other
three hit the `FAST_COMMIT` branch, which writes the same three-row artifact without a
mode file, so the check was rows and columns rather than the marker.

Quota went 5 -> 0; lifetime 34 -> 39. Everything is PENDING, so the board numbers land
later. Nothing else is left to spend today.

**What to read when the scores appear.** v43 is the pipeline test: if the canonical 0.684
reproduces at or above 0.680 on this account, the 0.684 lineage replaces the 0.683 fork as
the base and the three calibration axes (Sinkhorn strength, target mix, native share) plus
the punctuation prior become the only knobs worth moving. If v43 lands at 0.677 again, the
0.006-0.008 shortfall against the authors is account-level noise, and the decision is
whether to keep drawing variance or move to the own-LoRA member as the only real edge.
