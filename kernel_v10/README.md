# V10 — Official Prompt-Residual Readout

V10 keeps the scored V6 ensemble unchanged as its control: two-stage main model, 15% complementary stack, uncertain-row ATLAS, and Sinkhorn balancing. It adds a separate 5% auxiliary readout trained on official `train.csv` labels and hidden features from the existing `lora24_public` backbone.

The helper uses a fixed 512-dimensional sparse random projection, per-`prompt_id` feature centering, and multinomial logistic regression. It records a five-fold `GroupKFold` balanced-accuracy diagnostic, grouping by `prompt_id`; it does not tune on the public leaderboard. The helper's predictions are mixed at 5% before the existing final Sinkhorn step.

The general idea of separating prompt effects from model style is inspired by [Kirill Tymchishin's public experimental notebook](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature). That notebook explicitly says its experiment has no claimed official score. V10 does not use its private synthetic data or trained readout banks; the additional classifier is fitted locally in the Kaggle run using official labeled rows only.

The first corrected-input run reached the final blend after about 2,604 seconds, printed a prompt-grouped readout-only OOF balanced accuracy of 0.775411, then failed because the readout label columns and V6's main probability columns used different orders. The V10 builder was fixed to align columns by label name. Kaggle GPU notebook version 3 then completed in about 38 minutes, verified every required ensemble component, and produced a valid three-row preview. This run verifies execution and output formatting only, not leaderboard accuracy. Its `0.775411` OOF value is for the new readout alone, not the full V6 ensemble. Version 3 was submitted to Name That LLM as competition submission `57603012`; status is pending. No score improvement is claimed.

The notebook is generated from V6 by `build_v10.py`. It is an experimental hypothesis, not a claimed score improvement.
