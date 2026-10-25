import json
import os

def main():
    os.makedirs('kernel_v5', exist_ok=True)

    with open('kernel_v1/ntl_v1_baseline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Modify Cell 1 (Weights Configuration)
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if 'W_SVC = 1.0' in src:
            new_src = src.replace('W_SVC = 1.0', 'W_SVC = 0.75')
            new_src = new_src.replace(
                '"lora_qwen3_14b":  dict(w=0.4, twin_w=0.1, bs=8',
                '"lora_qwen3_14b":  dict(w=0.55, twin_w=0.15, bs=8'
            )
            new_src = new_src.replace(
                '"lora_qwen25_14b": dict(w=0.2, twin_w=0.1, bs=8',
                '"lora_qwen25_14b": dict(w=0.25, twin_w=0.10, bs=8'
            )
            new_src = new_src.replace('W_DEBERTA = 0.25', 'W_DEBERTA = 0.30')
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            print(f"Updated ensemble weights in cell {i}")

    # 2. Modify Cell 3 (Curated External Corpus, 5-View Stylometry with Lexical Dispersion)
    cell3_code = '''if not PUBLIC_SAMPLE:
    import string
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    VIEWS = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("small", 1, 3, 50000, 0.5),
        ("punct", 2, 4, 30000, 0.5)
    ]

    def shape_text(t):
        """Every run of letters becomes X or x, so only the rhythm is left."""
        return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(t))

    def small_words_text(t):
        """Keep the small words and the punctuation, and replace every other word by CONTENT."""
        return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                        for w in re.findall(r"\w+|[^\w\s]", str(t)))

    PUNCT_SET = set(string.punctuation).union({chr(0x2014), chr(0x2013), chr(0x2018), chr(0x2019), chr(0x201c), chr(0x201d)})

    def punct_text(t):
        """Extract punctuation rhythm: dashes, quotes, colons, brackets, and surrounding spacing."""
        return "".join(c for c in str(t) if c.isspace() or c in PUNCT_SET)

    pool = pd.read_parquet(find("pool_v4.parquet"))
    fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
    fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]

    try:
        ext_csv = find("passages_clean.csv")
        ext_df = pd.read_csv(ext_csv)
        ext_df = ext_df[ext_df["model"].isin(LABELS)]
        fit_texts += ext_df["text"].astype(str).tolist()
        fit_y = np.r_[fit_y, ext_df["model"].map({l: i for i, l in enumerate(LABELS)}).values]
        print(f"Attached {len(ext_df)} curated passages from passages_clean.csv", flush=True)
    except Exception as e:
        print(f"passages_clean.csv note: {e}", flush=True)

    A, B = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        if view == "punct":
            prep = punct_text
            analyzer = "char"
            opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                        max_features=max_features, dtype=np.float32)
        elif view == "shape":
            prep = shape_text
            analyzer = "char"
            opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                        max_features=max_features, dtype=np.float32)
        elif view == "small":
            prep = small_words_text
            analyzer = "word"
            opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                        max_features=max_features, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        elif view == "word":
            prep = lambda s: str(s)
            analyzer = "word"
            opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                        max_features=max_features, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        else:
            prep = lambda s: str(s)
            analyzer = "char"
            opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                        max_features=max_features, dtype=np.float32)
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        B.append(vec.transform([prep(t) for t in test.text]) * weight)

    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
    SVC_RAW = svc.decision_function(hstack(B, format="csr")).astype(np.float32)
    # Calibrated temperature scaling (1.25)
    SVC_P = softmax(SVC_RAW / 1.25)
    vote("svc", SVC_P, W_SVC)
    del A, B, svc, pool
    gc.collect()'''

    nb['cells'][3]['source'] = [line + '\n' for line in cell3_code.split('\n')]

    # 3. Modify Cell 5 (Multi-Scale Centred Twin Clustering)
    cell5_code = '''if not PUBLIC_SAMPLE:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(test) >= 2:
        Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
        D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
        np.fill_diagonal(D, 0)
        TREE = linkage(squareform(D, checks=False), method="average")
        groups_tight = fcluster(TREE, 0.82, criterion="distance")
        groups_broad = fcluster(TREE, 0.88, criterion="distance")
        del D
    else:
        groups_tight = np.ones(len(test), dtype=int)
        groups_broad = np.ones(len(test), dtype=int)
        TREE = None

    size_tight = np.bincount(groups_tight)[groups_tight]
    size_broad = np.bincount(groups_broad)[groups_broad]
    twin_rows = (size_broad >= 2) & (size_broad <= 18)
    print(f"Multi-scale topic clusters: {int((size_broad == 1).sum())} alone, {int(twin_rows.sum())} in clusters of 2 to 18", flush=True)

    for ckpt_name, mc in LORAS.items():
        ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))
        Z, heads = lora_features(ckpt, find_base(mc["base"]), test.text.tolist(), mc["bs"])
        vote(ckpt_name, softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), mc["w"])
        if twin_rows.any():
            Zc = Z.copy()
            # Multi-scale mean subtraction: 70% tight prompt cluster + 30% broad topic cluster
            for g in np.unique(groups_broad[twin_rows]):
                m_broad = Zc[groups_broad == g].mean(0)
                Zc[groups_broad == g] -= 0.30 * m_broad
            for g in np.unique(groups_tight[twin_rows]):
                m_tight = Zc[groups_tight == g].mean(0)
                Zc[groups_tight == g] -= 0.70 * m_tight
            centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
            vote(ckpt_name + " twin", softmax(centred), mc["twin_w"], rows=twin_rows)
        del Z
        gc.collect()'''

    nb['cells'][5]['source'] = [line + '\n' for line in cell5_code.split('\n')]

    # 4. Modify Cell 9 (Soft Typographic Priors + Multi-Scale Convex Shift Blend)
    cell9_code = '''def grouped_svc_shift(scores, group_ids, min_size=8, max_size=18):
    shift = np.zeros_like(scores, dtype=np.float64)
    counts = np.bincount(group_ids)[group_ids]
    valid = (counts >= min_size) & (counts <= max_size)
    for gid in np.unique(group_ids[valid]):
        idx = np.flatnonzero(group_ids == gid)
        center = np.mean(scores[idx], axis=0, dtype=np.float64)
        shift[idx] = center.mean() - center
    return shift, valid


if not PUBLIC_SAMPLE:
    def sinkhorn(P, iters=300):
        c = np.ones(K, dtype=np.float64)
        target = np.full(K, len(P) / K)
        for _ in range(iters):
            R = P * c
            R /= R.sum(axis=1, keepdims=True)
            mass = R.sum(axis=0)
            c *= target / np.maximum(mass, 1e-12)
        R = P * c
        return R / R.sum(axis=1, keepdims=True)

    def save_prediction(scores, filename):
        P = sinkhorn(softmax(scores))
        pred = np.asarray(LABELS)[P.argmax(axis=1)]
        sub = pd.DataFrame({'id': test.id.astype(str), 'model': pred})
        assert len(sub) == len(test) and sub.id.is_unique
        assert sub.model.isin(LABELS).all() and sub.id.notna().all()
        sub.to_parquet(filename, index=False)
        return pred

    original_scores = np.asarray(LOGIT, dtype=np.float64).copy()

    # --- Soft Typographic Prior Gating ---
    row_margin = np.sort(original_scores, axis=1)[:, -1] - np.sort(original_scores, axis=1)[:, -2]
    uncertainty = np.clip((2.0 - row_margin) / 2.0, 0.0, 1.0)

    curly_apos = chr(0x2019)
    curly_open_q = chr(0x201c)
    curly_close_q = chr(0x201d)
    em_dash = chr(0x2014)

    claude_glm_idx = [i for i, l in enumerate(LABELS) if l.startswith(("claude", "glm"))]
    gpt_idx = [i for i, l in enumerate(LABELS) if l.startswith("gpt")]

    typo_adjust = np.zeros_like(original_scores)
    for i, t in enumerate(test.text.values):
        s_text = str(t)
        u = uncertainty[i]
        if u > 0.05:
            if (curly_apos in s_text or curly_open_q in s_text or curly_close_q in s_text):
                typo_adjust[i, claude_glm_idx] -= 0.8 * u
            if (em_dash in s_text or " — " in s_text or " -- " in s_text):
                typo_adjust[i, claude_glm_idx] += 0.5 * u

    adjusted_scores = original_scores + typo_adjust

    topic_groups = fcluster(TREE, 0.90, criterion='distance')
    shift, accepted = grouped_svc_shift(SVC_RAW, topic_groups, 8, 18)
    gate = np.clip((1.9 - row_margin) / 1.5, 0.12, 1.0)
    gate[~accepted] = 0.0
    delta = W_SVC * shift * gate[:, None]

    # V5: Calibrated 3-Point Convex Mixture around the proven sweet spot (0.50 - 0.72)
    P_c50 = sinkhorn(softmax(adjusted_scores + 0.50 * delta))
    P_c62 = sinkhorn(softmax(adjusted_scores + 0.62 * delta))
    P_c72 = sinkhorn(softmax(adjusted_scores + 0.72 * delta))
    P_final = 0.30 * P_c50 + 0.50 * P_c62 + 0.20 * P_c72
    pred_final = np.asarray(LABELS)[P_final.argmax(axis=1)]

    sub = pd.DataFrame({'id': test.id.astype(str), 'model': pred_final})
    assert len(sub) == len(test) and sub.id.is_unique
    assert sub.model.isin(LABELS).all() and sub.id.notna().all()
    sub.to_parquet('submission.parquet', index=False)
    print('V5 Multi-Scale Centred Twin + Typo-Regularized submission.parquet written successfully!', flush=True)

    cache = dict(
        ids=test.id.astype(str).to_numpy(dtype=str),
        labels=np.asarray(LABELS, dtype=str),
        logit=adjusted_scores.astype(np.float32),
        svc_raw=SVC_RAW.astype(np.float32),
        topic_groups=np.asarray(topic_groups, dtype=np.int32),
        group_mask=np.asarray(accepted, dtype=np.uint8),
        gate=np.asarray(gate, dtype=np.float32),
        member_info=np.asarray(json.dumps([
            {'key': f'member_{i}', 'name': name, 'weight': MEMBER_WEIGHTS[name]}
            for i, name in enumerate(MEMBER_LOGP)
        ]))
    )
    for i, name in enumerate(MEMBER_LOGP):
        cache[f'member_{i}'] = MEMBER_LOGP[name]
    np.savez_compressed('v5_inference_cache.npz', **cache)

    print('complete topic groups', int(accepted.sum()), '/', len(test))
    print('group sizes', pd.Series(np.bincount(topic_groups)[topic_groups]).value_counts().sort_index().to_dict())
    print('loaded', list(MEMBER_LOGP), 'elapsed', round(time.time() - T0, 1), 'seconds')'''

    nb['cells'][9]['source'] = [line + '\n' for line in cell9_code.split('\n')]

    out_nb_path = 'kernel_v5/ntl_v5_multiscale_twins.ipynb'
    with open(out_nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=True)

    metadata = {
        "id": "sanjay0614/ntl-v5-multiscale-twins",
        "title": "NTL V5 Multiscale Twins",
        "code_file": "ntl_v5_multiscale_twins.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "nishantdahal/ntl-handwriting-blend",
            "prvsiyan/prvsiyans-extended-name-that-llm-dataset"
        ],
        "competition_sources": [
            "name-that-llm"
        ],
        "model_sources": [
            "qwen-lm/qwen-3/Transformers/14b-base/1",
            "qwen-lm/qwen2.5/Transformers/14b/1"
        ]
    }

    with open('kernel_v5/kernel-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print("Version 5 kernel files created successfully!")

if __name__ == '__main__':
    main()
