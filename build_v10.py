import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "kernel_v6" / "ntl_v6_public_complement_reproduction.ipynb"
OUTPUT_DIR = ROOT / "kernel_v10"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "ntl_v10_prompt_residual_readout.ipynb"

READOUT_INSERT = r'''        if key == 'lora24_public':
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.model_selection import GroupKFold
            from sklearn.preprocessing import normalize
            from sklearn.random_projection import SparseRandomProjection

            Z_train = features(TR, LORA_BS)
            assert Z_train.shape[1] == Z.shape[1] and len(Z_train) == len(y)
            projector = SparseRandomProjection(
                n_components=512, dense_output=True, random_state=42
            ).fit(Z_train)
            X_train = normalize(projector.transform(Z_train)).astype(np.float32)
            X_test = normalize(projector.transform(Z)).astype(np.float32)
            X_train = group_centre_dense(X_train, PROMPT.astype(str))
            X_test = group_centre_dense(X_test, G_TE)
            X_train = normalize(X_train).astype(np.float32)
            X_test = normalize(X_test).astype(np.float32)

            oof_pred = np.empty(len(y), dtype=np.int64)
            for fold_train, fold_valid in GroupKFold(n_splits=5).split(
                X_train, y, groups=PROMPT
            ):
                fold_model = LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    class_weight='balanced',
                    random_state=42,
                )
                fold_model.fit(X_train[fold_train], y[fold_train])
                oof_pred[fold_valid] = fold_model.predict(X_train[fold_valid])
            oof_balanced_accuracy = float(balanced_accuracy_score(y, oof_pred))
            print(
                f'OFFICIAL_PROMPT_GROUPED_READOUT_OOF '
                f'balanced_accuracy={oof_balanced_accuracy:.6f}',
                flush=True,
            )

            readout = LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight='balanced',
                random_state=42,
            ).fit(X_train, y)
            helper_probabilities = readout.predict_proba(X_test)
            assert list(readout.classes_) == list(range(K))
            assert np.isfinite(helper_probabilities).all()
            assert np.allclose(helper_probabilities.sum(1), 1.0, atol=1e-6)
            np.savez_compressed(
                '/kaggle/working/official_residual_readout.npz',
                ids=test.id.astype(str).to_numpy(dtype=str),
                labels=np.asarray(LABELS, dtype=str),
                probabilities=helper_probabilities.astype(np.float32),
                oof_balanced_accuracy=np.asarray(oof_balanced_accuracy),
            )
            del Z_train, X_train, X_test, oof_pred, readout, fold_model

'''

FINAL_BLEND_INSERT = r'''_readout = np.load('/kaggle/working/official_residual_readout.npz', allow_pickle=False)
assert _readout['ids'].tolist() == test.id.astype(str).tolist()
_readout_labels = _readout['labels'].tolist()
assert len(_readout_labels) == len(LABELS) and set(_readout_labels) == set(LABELS)
_readout_prob = _readout['probabilities'][:, [_readout_labels.index(label) for label in LABELS]]
assert _readout_prob.shape == P.shape
assert np.isfinite(_readout_prob).all() and np.allclose(_readout_prob.sum(1), 1.0, atol=1e-6)
P = 0.95 * P + 0.05 * _readout_prob
print(
    'V10_OFFICIAL_PROMPT_RESIDUAL_READOUT',
    f"weight=0.05 oof_balanced_accuracy={float(_readout['oof_balanced_accuracy']):.6f}",
    flush=True,
)

'''


def cell_text(cell):
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def set_cell_text(cell, text):
    source = cell.get("source", [])
    cell["source"] = text if isinstance(source, str) else text.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def generated_script_from_cell(cell):
    tree = ast.parse(cell_text(cell))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "write_text":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if "def score_lora_checkpoints" in argument.value:
                    return argument.value
    raise RuntimeError("Could not locate the embedded complementary-stack runner")


def replace_generated_script(cell, original_code, updated_code):
    if "Path('/kaggle/working/comp_stack.py').write_text(" not in cell_text(cell):
        raise RuntimeError("Unexpected complementary-stack writer cell")
    if "def score_lora_checkpoints" not in original_code:
        raise RuntimeError("Unexpected complementary-stack runner content")
    set_cell_text(
        cell,
        "from pathlib import Path\n"
        "Path('/kaggle/working/comp_stack.py').write_text("
        + repr(updated_code)
        + ")\n",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    script_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def score_lora_checkpoints" in cell_text(cell)
    ]
    if len(script_cells) != 1:
        raise RuntimeError(f"Expected one embedded runner cell, found {len(script_cells)}")
    script_cell = script_cells[0]
    original_script = generated_script_from_cell(script_cell)
    anchor = (
        "        Z = features(TE, LORA_BS)\n"
        "        add_member(key, softmax_rows(Z @ ro['plain_coef'].T + ro['plain_intercept']))\n"
    )
    if original_script.count(anchor) != 1:
        raise RuntimeError("Expected exactly one LoRA24 feature inference hook")
    updated_script = original_script.replace(
        anchor,
        "        Z = features(TE, LORA_BS)\n" + READOUT_INSERT + anchor.split("\n", 1)[1],
    )
    replace_generated_script(script_cell, original_script, updated_script)

    final_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "_atlas_prob[_atlas_rows]" in cell_text(cell)
        and "_before_balancing=P.copy()" in cell_text(cell)
    ]
    if len(final_cells) != 1:
        raise RuntimeError(f"Expected one final ATLAS blend cell, found {len(final_cells)}")
    final_cell = final_cells[0]
    final_source = cell_text(final_cell)
    marker = "_before_balancing=P.copy()"
    if final_source.count(marker) != 1:
        raise RuntimeError("Could not locate the post-ATLAS probability blend")
    set_cell_text(
        final_cell,
        final_source.replace(marker, FINAL_BLEND_INSERT + marker, 1),
    )

    title = """# Name That LLM — V10: Official Prompt-Residual Readout

V10 preserves V6's scored main ensemble, complementary stack, ATLAS path, and Sinkhorn balancing. It adds one independently fitted 5%-weight readout: the public `lora24_public` hidden features are randomly projected, centered within official prompt groups, and classified with multinomial logistic regression. The OOF diagnostic uses GroupKFold on `prompt_id`, so prompt siblings remain together.

The idea of separating prompt effects from model style is inspired by the public experimental notebook [0.639 | Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature). Its own notebook says that version has no claimed official score; its title is not treated as score evidence. V10 trains the auxiliary readout on official training features only; it does not use that notebook's private synthetic-data readout bank.

This is an experimental candidate, not a claimed improvement. Kaggle's three-row preview can verify execution and submission format only. V10 has not been submitted.
"""
    set_cell_text(notebook["cells"][0], title)
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
