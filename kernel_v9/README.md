# V9 — Augmented SVC + Complementary Stack

V9 is based on the V6 public complementary-stack + ATLAS reproduction. It preserves that ensemble and changes only its n-gram SVC cell:

- Add `passages_clean.csv` to the SVC training corpus.
- Add the punctuation-only character n-gram view.
- Keep an explicit word-token regex and apply the V2 SVC temperature of 1.25.

This is a targeted attempt to carry V2's strongest documented SVC changes into the complementary ensemble. It is not a proven improvement: the prior local SVC-only prompt-grouped experiment scored close to chance and did not validate the full ensemble. That CPU experiment is not used to claim a V9 score.

The notebook enables Kaggle GPU and retains V6's dependencies. Its Kaggle smoke execution on the three-row public preview verified that the actual model components execute and that the output is well-formed; it cannot validate accuracy. Notebook version 2 was submitted as competition submission `57597207`, which is pending a score.

The notebook is generated from the V6 source with `build_v9.py`.
