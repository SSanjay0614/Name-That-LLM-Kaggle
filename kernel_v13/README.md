# V13 — GLM/Kimi SVC Pool Augmentation

V13 uses V10 as its baseline and leaves all neural members, the official-only prompt-residual readout, blend weights, and Sinkhorn balancing unchanged. It adds the 1,516 CC BY 4.0 GLM-5.3, GLM-5.3-flash, and Kimi-K3 passages to the existing four-view n-gram SVC training pool.

A public GLM/Kimi pool-patch experiment scored 0.640 versus its 0.639 control, but it used a different 0.639 notebook. That finding motivates this controlled transfer test; it does not establish an expected V13 score. Dataset attribution: Daniel Andres Miranda, [NTL OpenRouter GLM and Kimi passages](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi), CC BY 4.0.

The Kaggle notebook's visible three-row fixture verifies execution and formatting only. It is not validation evidence. The full competition score must come from the hidden-test rerun.
