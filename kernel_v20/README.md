# NTL V20 V27 GLM Kimi SVC Pool

Parent: the public [NLM V27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) recipe, reproduced with pinned inputs by [Avazaku](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction). That author's official 0.651 score is a reference, not our result.

Keep the reproduced V27 recipe fixed and append the 1,516 CC BY 4.0 GLM-5.3, GLM-5.3-flash, and Kimi-K3 passages to its SVC pool only. Remove exact (model, text) duplicates against the V27 pool in Kaggle before fitting. The change is motivated by our prompt-grouped SVC-only V16 ablation (both sources +0.00333 BA) and full-pipeline V13 result (0.640 vs V10's 0.639); neither establishes a gain on the V27 stack.

The public reproduction recorded a visible Kaggle run of about 4 h 4 min on T4 x2, and its hidden-run time was not reported. This project run must remain within the live 9-hour notebook limit. It is a Kaggle cloud run; no local model fitting is performed.

The upstream notebook code is Apache 2.0 and the input datasets/models keep their own terms. The reproduced input inventory includes CC BY-SA 4.0 handwriting-blend data and Llama 3.1 Community License assets; preserve the original notices and source links. V20 adds only the CC BY 4.0 GLM/Kimi passage dataset.
