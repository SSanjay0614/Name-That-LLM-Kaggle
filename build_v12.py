import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "kernel_v10"
OUTPUT_DIR = ROOT / "kernel_v12"
SOURCE_NOTEBOOK = SOURCE_DIR / "ntl_v10_prompt_residual_readout.ipynb"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "ntl_v12_projection-ensemble.ipynb"

READOUT_REPLACEMENT = r'''        if key == 'lora24_public':
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.model_selection import GroupKFold
            from sklearn.preprocessing import normalize
            from sklearn.random_projection import SparseRandomProjection

            Z_train = features(TR, LORA_BS)
            assert Z_train.shape[1] == Z.shape[1] and len(Z_train) == len(y)
            projection_seeds = (42, 123, 2027)
            oof_probabilities = []
            helper_probabilities = np.zeros((len(Z), K), dtype=np.float32)

            for seed in projection_seeds:
                projector = SparseRandomProjection(
                    n_components=512, dense_output=True, random_state=seed
                ).fit(Z_train)
                X_train = normalize(projector.transform(Z_train)).astype(np.float32)
                X_test = normalize(projector.transform(Z)).astype(np.float32)
                X_train = group_centre_dense(X_train, PROMPT.astype(str))
                X_test = group_centre_dense(X_test, G_TE)
                X_train = normalize(X_train).astype(np.float32)
                X_test = normalize(X_test).astype(np.float32)

                seed_oof = np.zeros((len(y), K), dtype=np.float32)
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
                    assert np.array_equal(fold_model.classes_, np.arange(K))
                    seed_oof[fold_valid] = fold_model.predict_proba(X_train[fold_valid])

                readout = LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    class_weight='balanced',
                    random_state=42,
                ).fit(X_train, y)
                assert np.array_equal(readout.classes_, np.arange(K))
                seed_test = readout.predict_proba(X_test)
                assert np.isfinite(seed_test).all()
                assert np.allclose(seed_test.sum(1), 1.0, atol=1e-6)
                helper_probabilities += seed_test.astype(np.float32) / len(projection_seeds)
                oof_probabilities.append(seed_oof)
                print(f'V12_PROJECTION_COMPLETE seed={seed}', flush=True)
                del projector, X_train, X_test, seed_test, seed_oof, readout, fold_model

            mean_oof_probabilities = np.mean(oof_probabilities, axis=0)
            oof_balanced_accuracy = float(
                balanced_accuracy_score(y, mean_oof_probabilities.argmax(axis=1))
            )
            print(
                f'V12_PROMPT_GROUPED_ENSEMBLE_OOF '
                f'balanced_accuracy={oof_balanced_accuracy:.6f} '
                f'projections={len(projection_seeds)}',
                flush=True,
            )
            assert np.isfinite(helper_probabilities).all()
            assert np.allclose(helper_probabilities.sum(1), 1.0, atol=1e-6)
            np.savez_compressed(
                '/kaggle/working/official_residual_readout.npz',
                ids=test.id.astype(str).to_numpy(dtype=str),
                labels=np.asarray(LABELS, dtype=str),
                probabilities=helper_probabilities,
                oof_balanced_accuracy=np.asarray(oof_balanced_accuracy),
            )
            del Z_train, oof_probabilities, mean_oof_probabilities

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

    title_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
        and "Name That LLM" in cell_text(cell)
        and "V10" in cell_text(cell)
    ]
    if len(title_cells) != 1:
        raise RuntimeError(f"Expected one V10 title cell, found {len(title_cells)}")
    title = cell_text(title_cells[0])
    title = title.replace("V10", "V12", 1).replace(
        "one independently fitted 5%-weight readout",
        "one 5%-weight readout averaging three independently seeded sparse projections",
        1,
    )
    title = title.replace(
        "The OOF diagnostic uses GroupKFold on `prompt_id`, so prompt siblings remain together.",
        "The readout averages three 512-dimensional sparse projections (seeds 42, 123, "
        "and 2027). Its OOF diagnostic uses GroupKFold on `prompt_id`, so prompt siblings "
        "remain together. The 5% mixture weight and all other ensemble components match V10.",
    )
    set_cell_text(title_cells[0], title)

    script_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def score_lora_checkpoints" in cell_text(cell)
    ]
    if len(script_cells) != 1:
        raise RuntimeError(f"Expected one embedded runner cell, found {len(script_cells)}")
    script_cell = script_cells[0]
    source = generated_script_from_cell(script_cell)
    start_marker = "        if key == 'lora24_public':\n"
    end_marker = "\n        add_member(key, softmax_rows(Z @ ro['plain_coef'].T + ro['plain_intercept']))"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate V10 readout insertion boundaries")
    updated_script = source[:start] + READOUT_REPLACEMENT.rstrip() + source[end:]
    ast.parse(updated_script)
    replace_generated_script(script_cell, source, updated_script)

    blend_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "_readout = np.load('/kaggle/working/official_residual_readout.npz'" in cell_text(cell)
    ]
    if len(blend_cells) != 1:
        raise RuntimeError(f"Expected one final readout blend cell, found {len(blend_cells)}")
    blend = cell_text(blend_cells[0])
    if "P = 0.95 * P + 0.05 * _readout_prob" not in blend:
        raise RuntimeError("V10 baseline readout weight was not found")
    blend = blend.replace(
        "V10_OFFICIAL_PROMPT_RESIDUAL_READOUT",
        "V12_THREE_PROJECTION_PROMPT_RESIDUAL_READOUT",
        1,
    )
    set_cell_text(blend_cells[0], blend)

    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    metadata = json.loads((SOURCE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = "sanjay0614/ntl-v12-three-projection-residual-readout"
    metadata["title"] = "NTL V12 Three-Projection Residual Readout"
    metadata["code_file"] = OUTPUT_NOTEBOOK.name
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
