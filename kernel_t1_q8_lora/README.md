# kernel_t1_q8_lora — our own NTL member

Notebook: `ntl_q8_lora_pool.ipynb` (built from `../trainer/train_q8_lora.py` by
`../tools/make_trainer_nb.py`). Kaggle id `sanjay0614/ntl-v43-q8-lora-pool`, private, T4,
no internet, dataset `sanjay0614/ntl-pool-v4`, model `qwen-lm/qwen-3/Transformers/8b-base/1`.

## Why a trained member and not another fork

The public stacks all sit at 0.680–0.684 while the leaderboard top is 0.765. Independent forks
move blend weights and land inside ±0.009 of each other, so the remaining gap cannot be
calibration — it needs a member the public stack does not already have. Training on the open
pool is the only lever left that does not require paid generation.

## What is different from the public LoRA members

| lever | public practice | here |
|---|---|---|
| class balance | plain CE on the pool prior, fixed by Sinkhorn at inference | `WeightedRandomSampler` with 1/count weights, so `claude-opus-5` (2,338 rows) gets the same gradient budget as `gpt-5.6-luna` (6,736) |
| validation | often a random split or a component-only metric | **prompt-grouped** holdout, 12 % of pool prompts and 15 % of official-train prompts, evaluated separately so the competition distribution is visible |
| reporting | single score | per-class recall and the top-3 confusions per class, so a blend can decide *which* classes to trust this member on |
| pooling | layer-24 hidden states + trained head | mean-pooled final hidden state + fp32 linear head |

## Data

`pool_v4.parquet` — 75,783 labelled passages, 18 classes, 8,278 prompt groups, 25 public
sources, 30–298 words. `train.csv` — the official 2,699 rows. Exact-text duplicates are dropped
and the two corpora are split independently, then their training parts are concatenated.

## Run shape

- Qwen3-8B, 4-bit NF4 + double quant, LoRA r=16 α=32 on all attention and MLP projections,
  gradient checkpointing, fp16 compute (T4 has no bf16).
- micro-batch 4 × accum 8 = effective 32, `max_len` 384, dynamic per-batch padding.
- lr 2e-4 cosine, 3 % warmup, 2 epochs planned, **time-guarded at 7.2 h**.
- Evaluation at 25/50/75/100 % of the step budget (and again in epoch 2), best adapter saved on
  pool-holdout balanced accuracy.

## Artifacts (kernel output, `/kaggle/working/ntl_q8_lora`)

| file | use |
|---|---|
| `best/adapter_model.safetensors`, `best/adapter_config.json` | LoRA weights (PEFT) |
| `best/head.pt` | the 18-way linear head — **required**, it is outside the PEFT adapter |
| `best/tokenizer*`, `best/config.json` | reload without the base tokenizer |
| `config_used.json` | base path, hyper-parameters, label order |
| `history.json`, `diagnostics.json` | per-eval and per-class results |
| `pred_pool_va.parquet`, `pred_official_va.parquet` | full 18-column probabilities on both holdouts |
| `DONE` | elapsed hours |

## Reusing it

The adapter is only useful if it beats the equivalent public member, so the gate is
`diagnostics.json`: pool-holdout balanced accuracy must be clearly above the 0.6603-style
component numbers the public scorers report before it earns a slot. To deploy it, download the
output, `kaggle datasets create` it, attach that dataset plus the same Qwen3-8B source to a
candidate kernel, and reconstruct `mean-pooled final hidden state → head.pt`.

## Known risks

1. **Runtime.** ~1 s per micro-batch implies ~5.4 h per epoch on one T4; the guard stops at
   7.2 h, so expect epoch 1 plus part of epoch 2. The LR schedule assumes 2 epochs, so a cut at
   7.2 h leaves the tail of the cosine unused.
2. **Rank ceiling.** One epoch over 76 k rows is less compute than the public members got. If
   the holdout lands near 0.60 this member is not worth a slot.
3. **No test-file dependency.** `competition_sources` is empty, so the notebook trains the same
   way in a commit run and a rerun and cannot be tripped by a missing `test.csv`.
