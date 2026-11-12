# %% [markdown]
# # NTL V43 - Qwen3-8B QLoRA on the 75.8k open pool
#
# **Why this notebook exists.** Every public stack on this competition converges on
# 0.680-0.684 no matter which blend weight it moves, while the leaderboard top is 0.765. That
# gap is not calibration, so the only lever left on open data is a *member* that the public
# stack does not already contain.
#
# Three deliberate differences from the public LoRA members:
#
# 1. **Class-balanced sampling.** The metric is balanced accuracy, but the pool is imbalanced
#    (2,338 rows for `claude-opus-5` vs 6,736 for `gpt-5.6-luna`). A plain cross-entropy run
#    learns the pool prior and then leans on Sinkhorn to fix it at inference. Here every class
#    is drawn with equal probability during training, so the rare classes get real gradient.
# 2. **Prompt-grouped validation.** Holdout prompts are excluded from training entirely,
#    because the same prompt leaks topic across all 18 models.
# 3. **Honest reporting.** Per-class recall and the top confusion pairs are written out, so a
#    downstream blend can see *which* classes this member is worth attaching for.
#
# Training is time-guarded and checkpoints after every evaluation, so a kill at the 9 h limit
# still leaves a usable adapter.

# %%
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

T0 = time.time()
DATA = "/kaggle/input/ntl-pool-v4"
OUT = "/kaggle/working/ntl_q8_lora"


def find_data(name):
    """Resolve a dataset file under whichever mount layout this session gets.

    The first run failed with /kaggle/input/ntl-pool-v4/pool_v4.parquet missing: the
    dataset was mounted one level deeper than the default layout. Try the two known
    layouts, then a bounded glob, so a mount-layout change cannot kill the run again.
    """
    for root in (DATA, "/kaggle/input/datasets/sanjay0614/ntl-pool-v4"):
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under /kaggle/input; top level: "
                            + str(sorted(os.listdir("/kaggle/input"))))
os.makedirs(OUT, exist_ok=True)

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
LAB2I = {l: i for i, l in enumerate(LABELS)}

MAX_LEN = 384
MICRO_BATCH = 4
ACCUM = 8
EPOCHS = 2
LR = 2e-4
WARMUP = 0.03
TIME_BUDGET_S = 7.2 * 3600
SEED = 20261111

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count(), flush=True)


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find_qwen_base():
    """Resolve the attached Qwen3-8B base under whichever mount layout this session gets."""
    patterns = [
        "/kaggle/input/models/qwen-lm/qwen-3/*/8b-base/*",
        "/kaggle/input/qwen-3/*/8b-base/*",
        "/kaggle/input/qwen-3/transformers/8b-base/*",
        "/kaggle/input/*qwen*8b*base*/*",
    ]
    seen = []
    for pat in patterns:
        for cand in sorted(glob.glob(pat)):
            if os.path.isfile(os.path.join(cand, "config.json")):
                seen.append(cand)
    if not seen:  # last resort: one shallow walk of the mount root
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
            if depth > 6:
                dirs[:] = []
                continue
            if "config.json" in files and "tokenizer.json" in files:
                seen.append(root)
        seen = [d for d in seen if "qwen" in d.lower()]
    if not seen:
        raise FileNotFoundError("Qwen3-8B base not found; mounted inputs: "
                                + str(sorted(os.listdir("/kaggle/input"))))
    return sorted(seen)[0]


MODEL_PATH = find_qwen_base()
import transformers
log(f"transformers {transformers.__version__} | base model: {MODEL_PATH}")


def install_offline_wheel(pattern):
    """Install a wheel shipped in an attached dataset; the session has no internet.

    The second run reached the model and died on `pip install -U bitsandbytes>=0.46.1`:
    the image does not ship bitsandbytes and there is no index to fetch it from. The
    public stacks attach dhruvtibarewal/nlm-offline-wheels for exactly this reason, so
    the wheel is resolved from /kaggle/input instead.
    """
    wheels = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not wheels:
        log(f"offline wheel {pattern}: NOT FOUND")
        return None
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--no-deps", "--disable-pip-version-check", "-q", wheels[0]],
                   check=True)
    log(f"offline wheel {pattern}: {wheels[0]}")
    return wheels[0]


install_offline_wheel("bitsandbytes-*.whl")

# %% [markdown]
# ## Data: pool + official train, deduplicated, split by prompt group
#
# The pool adds breadth (25 sources, 8,278 prompts); the official file adds the exact
# distribution the leaderboard samples from. They are split **separately** so each keeps its own
# honest holdout, then the two training parts are concatenated.

# %%
def load_corpus():
    pool = pd.read_parquet(find_data("pool_v4.parquet"))
    pool = pool[["prompt_key", "text", "model"]].copy()
    pool["prompt_key"] = "pool:" + pool["prompt_key"].astype(str)
    pool["origin"] = "pool"

    off = pd.read_csv(find_data("train.csv"), dtype={"id": str})
    off = off[["prompt_id", "text", "model"]].rename(columns={"prompt_id": "prompt_key"})
    off["prompt_key"] = "off:" + off["prompt_key"].astype(str)
    off["origin"] = "official"

    df = pd.concat([pool, off], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["model"].isin(LABELS)]
    df = df[df["text"].str.len() >= 40]
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    log(f"corpus rows {before} -> {len(df)} after exact-text dedup")
    df["y"] = df["model"].map(LAB2I).astype(np.int64)
    return df


def group_split(df, frac, seed):
    """Hold out whole prompts, never passages."""
    prompts = df["prompt_key"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    n_val = max(1, int(round(len(prompts) * frac)))
    val = set(prompts[:n_val])
    mask = df["prompt_key"].isin(val)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


corpus = load_corpus()
pool = corpus[corpus.origin == "pool"]
official = corpus[corpus.origin == "official"]
pool_tr, pool_va = group_split(pool, 0.12, SEED)
off_tr, off_va = group_split(official, 0.15, SEED + 1)
train = pd.concat([pool_tr, off_tr], ignore_index=True)
log(f"train {len(train)} | pool holdout {len(pool_va)} | official holdout {len(off_va)}")
log("train class counts min/max = %d/%d" % (train.groupby('y').size().min(),
                                            train.groupby('y').size().max()))
for name, part in (("pool_va", pool_va), ("off_va", off_va)):
    missing = sorted(set(range(18)) - set(part["y"].unique()))
    log(f"{name}: {len(part)} rows, classes missing {missing}")

# %% [markdown]
# ## Model: 4-bit base + LoRA + mean-pooled head

# %%
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base = AutoModel.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.float16, trust_remote_code=False,
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base, lora)
try:
    model.gradient_checkpointing_enable()
except Exception as exc:  # already enabled by prepare_model_for_kbit_training
    log(f"gradient checkpointing already active: {exc}")
model.print_trainable_parameters()

hidden = model.config.hidden_size
head = nn.Linear(hidden, len(LABELS)).to("cuda:0")
nn.init.normal_(head.weight, std=0.02)
nn.init.zeros_(head.bias)

params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
log(f"hidden {hidden} | trainable tensors {len(params)} | labels {len(LABELS)}")

# %% [markdown]
# ## Datasets and class-balanced sampling

# %%
class Passages(Dataset):
    def __init__(self, frame):
        self.text = frame["text"].tolist()
        self.y = frame["y"].to_numpy(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.text[i], int(self.y[i])


class PadCollate:
    def __call__(self, batch):
        texts = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], ys


train_ds = Passages(train)
counts = np.bincount(train["y"].to_numpy(), minlength=len(LABELS)).astype(np.float64)
class_w = 1.0 / np.clip(counts, 1.0, None)
sample_w = class_w[train["y"].to_numpy()]
sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                               num_samples=len(train_ds), replacement=True)
train_dl = DataLoader(train_ds, batch_size=MICRO_BATCH, sampler=sampler,
                      collate_fn=PadCollate(), num_workers=2, drop_last=True,
                      pin_memory=True)

steps_per_epoch = math.ceil(len(train_dl) / ACCUM)
total_steps = steps_per_epoch * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, int(total_steps * WARMUP), total_steps)
log(f"steps/epoch {steps_per_epoch} | total {total_steps} | effective batch {MICRO_BATCH * ACCUM}")


@torch.no_grad()
def predict(frame, batch=16):
    model.eval()
    ds = Passages(frame)
    probs, ys = [], []
    for start in range(0, len(ds), batch):
        chunk = [(ds.text[i], ds.y[i]) for i in range(start, min(start + batch, len(ds)))]
        ids, mask, y = PadCollate()(chunk)
        out = model(input_ids=ids.to("cuda:0"), attention_mask=mask.to("cuda:0"),
                    output_hidden_states=True)
        h = out.hidden_states[-1]
        m = mask.to("cuda:0").unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        logits = head(pooled.float())
        probs.append(torch.softmax(logits, -1).float().cpu().numpy())
        ys.append(y.numpy())
    model.train()
    return np.concatenate(probs), np.concatenate(ys)


def balanced_accuracy(probs, ys):
    pred = probs.argmax(1)
    rec = []
    per_class = {}
    for c in range(len(LABELS)):
        sel = ys == c
        if sel.sum() == 0:
            continue
        r = float((pred[sel] == c).mean())
        rec.append(r)
        per_class[LABELS[c]] = round(r, 4)
    return float(np.mean(rec)), per_class


history = []
best = -1.0
best_dir = os.path.join(OUT, "best")


def evaluate(tag):
    global best
    rows = {}
    for name, part in (("pool_va", pool_va), ("official_va", off_va)):
        if len(part) == 0:
            continue
        probs, ys = predict(part)
        ba, per_class = balanced_accuracy(probs, ys)
        rows[name] = {"balanced_accuracy": round(ba, 5), "per_class_recall": per_class,
                      "n": int(len(part))}
        log(f"{tag} {name}: balanced accuracy {ba:.4f}")
    pool_ba = rows.get("pool_va", {}).get("balanced_accuracy", 0.0)
    if pool_ba > best:
        best = pool_ba
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        torch.save(head.state_dict(), os.path.join(best_dir, "head.pt"))
        rows["saved"] = best_dir
        log(f"{tag} new best ({best:.4f}) -> {best_dir}")
    history.append({"tag": tag, "elapsed_s": round(time.time() - T0, 1), **rows})
    with open(os.path.join(OUT, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    return rows


# %% [markdown]
# ## Training loop
#
# Evaluation happens at 25/50/75/100% of each epoch, so the time guard can stop the run with a
# usable adapter even if one epoch does not fit.

# %%
model.train()
global_step = 0
micro = 0
opt.zero_grad(set_to_none=True)
running = []
eval_at = {int(total_steps * f) for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)}
stopped = False

for epoch in range(EPOCHS):
    for ids, mask, ys in train_dl:
        t0 = time.time()
        try:
            out = model(input_ids=ids.to("cuda:0"), attention_mask=mask.to("cuda:0"),
                        output_hidden_states=True)
            h = out.hidden_states[-1]
            m = mask.to("cuda:0").unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            logits = head(pooled.float())
            loss = nn.functional.cross_entropy(logits, ys.to("cuda:0"))
            if not torch.isfinite(loss):
                log("non-finite loss; skipping micro-batch")
                opt.zero_grad(set_to_none=True)
                continue
            (loss / ACCUM).backward()
        except torch.cuda.OutOfMemoryError:
            log("CUDA OOM on a micro-batch; skipping it")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        running.append(float(loss.detach()))
        micro += 1
        if micro % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 25 == 0:
                rate = (time.time() - T0) / max(global_step, 1)
                left = (total_steps - global_step) * rate
                log(f"step {global_step}/{total_steps} loss {np.mean(running[-100:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} eta {left / 3600:.1f}h")
            if global_step in eval_at:
                evaluate(f"ep{epoch + 1}@{global_step}")
            if time.time() - T0 > TIME_BUDGET_S:
                log("time budget reached; stopping training")
                stopped = True
                break
        if time.time() - t0 > 30:
            log(f"slow micro-batch {time.time() - t0:.1f}s")
    if stopped:
        break

log(f"training finished at step {global_step}/{total_steps} in {(time.time() - T0) / 3600:.2f}h")
if os.path.isdir(best_dir):
    head.load_state_dict(torch.load(os.path.join(best_dir, "head.pt"), map_location="cuda:0"))
    log("restored best head for final diagnostics")

# %% [markdown]
# ## Final diagnostics

# %%
final = {}
for name, part in (("pool_va", pool_va), ("official_va", off_va)):
    if len(part) == 0:
        continue
    probs, ys = predict(part)
    ba, per_class = balanced_accuracy(probs, ys)
    pred = probs.argmax(1)
    conf = {}
    for c in range(len(LABELS)):
        sel = ys == c
        if sel.sum() == 0:
            continue
        wrong = pred[sel]
        wrong = wrong[wrong != c]
        if len(wrong):
            vals, cnts = np.unique(wrong, return_counts=True)
            order = np.argsort(-cnts)[:3]
            conf[LABELS[c]] = {LABELS[vals[i]]: int(cnts[i]) for i in order}
    final[name] = {"balanced_accuracy": round(ba, 5), "per_class_recall": per_class,
                   "top_confusions": conf, "n": int(len(part))}
    pd.DataFrame({"text_id": np.arange(len(part)), "y": ys,
                  "pred": pred,
                  **{f"p_{LABELS[i]}": probs[:, i].round(5) for i in range(len(LABELS))}}
                 ).to_parquet(os.path.join(OUT, f"pred_{name}.parquet"), index=False)
    log(f"final {name}: {ba:.4f}")

config = {
    "base_model": MODEL_PATH, "max_len": MAX_LEN, "micro_batch": MICRO_BATCH,
    "accum": ACCUM, "epochs": EPOCHS, "lr": LR, "seed": SEED,
    "labels": LABELS, "pool_rows": int((corpus.origin == "pool").sum()),
    "train_rows": int(len(train)), "pool_valid_rows": int(len(pool_va)),
    "official_valid_rows": int(len(off_va)), "steps_done": int(global_step),
    "class_balanced_sampler": True, "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
}
with open(os.path.join(OUT, "config_used.json"), "w") as fh:
    json.dump(config, fh, indent=2)
with open(os.path.join(OUT, "diagnostics.json"), "w") as fh:
    json.dump(final, fh, indent=2)
open(os.path.join(OUT, "DONE"), "w").write(json.dumps({"elapsed_h": round((time.time() - T0) / 3600, 2)}))
log("artifacts: " + ", ".join(sorted(os.listdir(OUT))))
