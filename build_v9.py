import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "kernel_v6" / "ntl_v6_public_complement_reproduction.ipynb"
OUTPUT_DIR = ROOT / "kernel_v9"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "ntl_v9_augmented_complement_stack.ipynb"

SVC_CELL = r'''import string
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
PUNCT_SET = set(string.punctuation).union(
    {chr(0x2014), chr(0x2013), chr(0x2018), chr(0x2019), chr(0x201c), chr(0x201d)}
)
VIEWS = [
    ("char", 3, 5, 250000, 1.0),
    ("word", 1, 3, 180000, 0.7),
    ("shape", 2, 6, 60000, 0.5),
    ("small", 1, 3, 50000, 0.5),
    ("punct", 2, 4, 30000, 0.5),
]


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda match: "X" if match[0].isupper() else "x", str(text))


def small_words_text(text):
    return " ".join(
        word.lower() if word.lower() in SMALL_WORDS or not word.isalnum() else "CONTENT"
        for word in re.findall(r"\w+|[^\w\s]", str(text))
    )


def punctuation_text(text):
    return "".join(char for char in str(text) if char.isspace() or char in PUNCT_SET)


pool = pd.read_parquet(find("pool_v4.parquet"))
fit_texts = train.text.fillna("").astype(str).tolist() + pool.text.fillna("").astype(str).tolist()
pool_labels = pool.model.map({label: index for index, label in enumerate(LABELS)})
assert pool_labels.notna().all(), "Public pool contains an unknown model label"
fit_y = np.r_[y, pool_labels.to_numpy(dtype=np.int64)]

ext_df = pd.read_csv(find("passages_clean.csv"))
ext_df = ext_df[ext_df["model"].isin(LABELS)].copy()
assert len(ext_df), "Curated passage dataset contains no known model labels"
fit_texts += ext_df.text.fillna("").astype(str).tolist()
fit_y = np.r_[fit_y, ext_df.model.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=np.int64)]
print(f"Added {len(ext_df)} curated passages to the SVC training pool", flush=True)

train_matrices, test_matrices = [], []
for view, lower, upper, max_features, weight in VIEWS:
    if view == "shape":
        prepare, analyzer = shape_text, "char"
    elif view == "small":
        prepare, analyzer = small_words_text, "word"
    elif view == "punct":
        prepare, analyzer = punctuation_text, "char"
    else:
        prepare, analyzer = (lambda text: str(text)), view

    options = dict(
        analyzer=analyzer,
        ngram_range=(lower, upper),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=max_features,
        dtype=np.float32,
    )
    if analyzer == "word":
        options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"

    vectorizer = TfidfVectorizer(**options)
    train_matrices.append(
        vectorizer.fit_transform([prepare(text) for text in fit_texts]) * weight
    )
    test_matrices.append(
        vectorizer.transform(test.text.fillna("").astype(str).tolist()) * weight
    )
    print(f"SVC view {view}: {train_matrices[-1].shape[1]} features", flush=True)

svc = LinearSVC(
    C=0.5,
    random_state=42,
    max_iter=5000,
    class_weight="balanced",
).fit(hstack(train_matrices, format="csr"), fit_y)
svc_raw = svc.decision_function(hstack(test_matrices, format="csr")).astype(np.float32)
vote("svc", softmax(svc_raw / 1.25), W_SVC)
del train_matrices, test_matrices, svc, pool, ext_df, fit_texts, fit_y
gc.collect()
'''


def set_cell_source(cell, text):
    old_source = cell.get("source", [])
    if isinstance(old_source, str):
        cell["source"] = text
    else:
        cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    matches = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "SMALL_WORDS = set(" in "".join(cell.get("source", []))
        and "LinearSVC" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one SVC cell, found {len(matches)}")
    set_cell_source(matches[0], SVC_CELL)

    intro = """# Name That LLM — V9: Augmented SVC + Complementary Stack

V9 preserves the V6 public complementary-stack + ATLAS ensemble and changes its SVC component to use the V2 curated-passage corpus, punctuation view, calibrated temperature, and explicit word-token pattern. This is an experiment, not a claimed score improvement.

The public test preview contains only three rows and can verify execution and output format, not accuracy. V9 has not been submitted to the competition.

Source base: [Fleongg's public complementary notebook](https://www.kaggle.com/code/fleongg/ntl-part-two-complement). The public source's reported score is not a score for this candidate.
"""
    set_cell_source(notebook["cells"][0], intro)
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
