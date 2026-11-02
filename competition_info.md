# Name That LLM

Competition page: <https://www.kaggle.com/competitions/name-that-llm>  
Rules: <https://www.kaggle.com/competitions/name-that-llm/rules>

## Competition at a glance

- **Goal:** Identify which of 18 candidate language models generated each text passage.
- **Metric:** Balanced accuracy across the 18 classes, so each class contributes equally through its recall. Project notes also describe Sinkhorn/optimal-transport balancing to match the test prediction distribution to the competition's uniform class prior.
- **Submission type:** Kaggle code competition. The notebook performs offline inference and produces the competition's required prediction artifact.
- **Known constraints:** The project README records an offline code competition with two NVIDIA T4 GPUs and a January 14, 2027 deadline. Verify the live page for current deadlines, runtime limits, daily submission quota, exact test/output schema, GPU availability, and rules before every experiment or submission.
- **Leakage caution:** Keep prompt groups and other related passages together in validation folds. Do not use hidden test labels or public leaderboard feedback as training labels.

## Our approach

1. **Search for high-scoring public work.** Review public Kaggle notebooks, GitHub repositories, papers, and discussion posts. Record the score's evaluation source, validation method, submission ID/version if available, data/model dependencies, and license. Separate a verified public leaderboard score from a CV score or smoke-test result.
2. **Check licenses and data provenance.** Inspect each public notebook, corpus, model, and checkpoint before reuse. Reuse code and assets only when the license and competition rules permit it; retain attribution and provenance. If terms are unclear, use the research findings as inspiration and implement the method independently.
3. **Build from the strongest lawful baseline.** Reproduce the available strong pipeline in Kaggle, verify all required offline inputs, and confirm the full hidden test path and output format. Keep the prior best-scoring candidate as a control.
4. **Fine-tune and adapt with controlled experiments.** Test one well-motivated change at a time—for example, prompt-grouped training data, stylometric features, embeddings, calibration, or class balancing. Use grouped out-of-fold validation that reflects prompt/domain shifts; do not trust random-split results where related prompts can leak across folds.
5. **Validate before spending an attempt.** Confirm the actual competition rerun processes all test rows, emits legal class predictions in the expected schema, and finishes within limits. A three-row preview may validate packaging only; it is not evidence of full-test accuracy.
6. **Track lineage and results.** Record the parent version, exact code/data/model changes, licenses, validation method, OOF metric, Kaggle kernel version, submission reference, and public score. Keep negative results and avoid tuning narrowly to a noisy leaderboard.

The aim is to adapt strong public approaches responsibly and measure whether each change improves this competition's balanced-accuracy score—not to assume a public result will transfer unchanged.

## Build, validate, and submit

1. Check the live rules, remaining submission quota, deadline, and current best/control score:

   ```powershell
   kaggle competitions submission-limits name-that-llm
   kaggle competitions submissions name-that-llm
   ```

2. Push the candidate notebook to Kaggle:

   ```powershell
   kaggle kernels push -p .\name_that_llm\kernel_v15
   ```

   Substitute the target version folder. Then inspect the new kernel run:

   ```powershell
   kaggle kernels status <owner>/<kernel-name>
   kaggle kernels logs <owner>/<kernel-name>
   ```

3. Confirm the completed competition rerun processed the full test set and created a valid prediction artifact—not just a visible preview or smoke output.
4. Submit that output:

   ```powershell
   kaggle competitions submit name-that-llm `
     -f <submission-file> `
     -k <owner>/<kernel-name> `
     -v <kernel-version> `
     -m "Short description of the tested candidate"
   ```

   Replace placeholders with the actual output filename and completed Kaggle kernel version. Do not submit a partial preview as a full prediction.
5. Check the accepted submission and score:

   ```powershell
   kaggle competitions submissions name-that-llm
   ```

   Record the submission reference, score, and lessons in the project README/version notes.

## Git and project workflow

- Keep generated predictions, checkpoints, caches, secrets, and scratch data out of Git; follow `.gitignore`.
- After Kaggle submissions, provide Git commands for the user to inspect and run. Do not execute Git/GitHub commands unless explicitly asked.
- Do not add a Copilot co-author trailer to suggested commit commands.
- If these notes differ from the live Kaggle competition rules, follow the live rules.

## Current project status

The project README records V10 as the best confirmed public score at 0.639. V13–V15 have pending submissions recorded there; check Kaggle before updating their status. See [the project README](./README.md).
