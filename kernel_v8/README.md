# V8 Candidate — V2 with a Conservative Grouped-Shift Blend

V8 is copied from the best-scoring local recipe, V2 (public score 0.623). It changes only the final grouped-SVC-shift calibration:

- V2: 40% at strength 0.65, 60% at strength 0.80.
- V8: 30% at strength 0.50, 50% at strength 0.62, 20% at strength 0.72.

The ensemble weights, augmented SVC data and features, centred-twin path, margin gate, grouping, and Sinkhorn balancing are otherwise unchanged. The conservative range is motivated by the experiment notes' reported shift-strength ablations; this isolated V8 score will determine whether that transfers to the 0.623 V2 setup.

The Kaggle public-preview smoke run completed and wrote a valid `submission.parquet`. The preview is only three rows, so the notebook takes its explicit small-test fallback; V8's preview artifact is byte-identical to V2's. This verifies packaging/output only, not the modified full-test shift blend or its accuracy.

V8 was submitted using Kaggle notebook version 1 as competition submission `57593259` and scored **0.623**, tying V2. The preview output itself does not validate full-test accuracy.
