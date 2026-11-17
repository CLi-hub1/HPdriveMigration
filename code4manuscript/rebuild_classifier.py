"""Rebuild a local four-label DistilBERT classifier with independent grouped folds."""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from config import CLASSIFIER_WORK
ROOT = CLASSIFIER_WORK
ROOT.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": str(ROOT / "hf_cache"),
    "HF_HUB_CACHE": str(ROOT / "hf_cache" / "hub"),
    "HF_ASSETS_CACHE": str(ROOT / "hf_cache" / "assets"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "TOKENIZERS_PARALLELISM": "false",
})
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

LABELS = ["Technological/Infrastructural", "Institutional", "Behavioral/Cultural", "Nature-Based"]
MODEL_ID = "distilbert/distilbert-base-uncased"
SEED = 20261020
PARAMS = {"learning_rate": 1e-5, "epochs": 5, "batch_size": 8, "max_length": 512}


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(text):
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", str(text)).lower()).strip()


def text_join(action, description, title=""):
    parts = [str(x).strip() for x in (action, description) if str(x).strip()]
    return ". ".join(parts) if parts else str(title).strip()


def audit():
    df = pd.read_csv(ROOT / "human_labels.csv", keep_default_na=False)
    assert set(LABELS).issubset(df.columns)
    assert np.isin(df[LABELS].values, [0, 1]).all(), "Nonbinary or missing human labels"
    df["text"] = [text_join(a, b) for a, b in zip(df["Action"], df["Action description"])]
    df["text_key"] = df["text"].map(normalize)
    df["merge_key"] = df["merge"].map(normalize)
    df["valid_text"] = df.text_key.str.len() > 0
    df["encoding_warning"] = df["text"].str.contains("\ufffd|ï¿½|â€™|â€“|Ã", regex=True)
    df["label_key"] = df[LABELS].astype(str).agg("|".join, axis=1)
    # Link reporting accounts and identical source/preprocessed texts before splitting.
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for column in ["Account number", "text_key", "merge_key"]:
        seen = {}
        for idx, value in enumerate(df[column].astype(str)):
            if not value:
                continue
            if value in seen:
                union(idx, seen[value])
            else:
                seen[value] = idx
    df["group"] = [find(i) for i in range(len(df))]
    conflict_raw = set(df.loc[df.valid_text].groupby("text_key").label_key.nunique().loc[lambda s: s > 1].index)
    conflict_merge = set(df.loc[df.merge_key.ne("")].groupby("merge_key").label_key.nunique().loc[lambda s: s > 1].index)
    df["conflicting_label"] = df.text_key.isin(conflict_raw) | df.merge_key.isin(conflict_merge)
    df["keep_train"] = df.valid_text & ~df.conflicting_label
    duplicates = df.loc[df.keep_train].duplicated("text_key")
    df.loc[duplicates.index[duplicates], "keep_train"] = False
    df.to_csv(ROOT / "label_audit_rows.csv", index=False, encoding="utf-8-sig")
    df.loc[df.conflicting_label].to_csv(ROOT / "conflicting_labels.csv", index=False, encoding="utf-8-sig")
    train = df.loc[df.keep_train].copy().reset_index(drop=True)
    assert train.text_key.is_unique
    assert train.group.nunique() >= 5, "Fewer than 5 independent city/text components"
    train["fold"] = -1
    groups = train.group.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(train, groups=groups))
    for fold, (ti, vi) in enumerate(splits):
        assert not set(groups[ti]) & set(groups[vi])
        assert not set(train.loc[ti, "Account number"]) & set(train.loc[vi, "Account number"])
        assert not set(train.loc[ti, "text_key"]) & set(train.loc[vi, "text_key"])
        train.loc[vi, "fold"] = fold
        assert (train.loc[ti, LABELS].sum() > 0).all(), "A training fold lacks a positive label"
    assert (train.fold >= 0).all()
    train.to_csv(ROOT / "training_rows.csv", index=False, encoding="utf-8-sig")
    summary = {
        "original_rows": len(df), "raw_text_empty_rows": int((~df.valid_text).sum()),
        "original_merge_empty_rows": int(df["merge"].eq("").sum()),
        "encoding_warning_rows": int(df.encoding_warning.sum()),
        "conflicting_label_rows_excluded": int(df.conflicting_label.sum()),
        "exact_same_text_duplicates_excluded": int(duplicates.sum()),
        "training_unique_texts": len(train),
        "accounts": int(df["Account number"].nunique()),
        "connected_groups_after_exclusions": int(train.group.nunique()),
        "largest_group_sizes": train.groupby("group").size().nlargest(10).to_dict(),
        "fold_sizes": train.groupby("fold").size().to_dict(),
        "fold_label_counts": train.groupby("fold")[LABELS].sum().to_dict("index"),
        "text_input": "Original Action + '. ' + Action description; no augmentation or lemmatization",
        "deduplication": "One copy per exact normalized original text; all inconsistent text-label identities excluded pending human adjudication",
        "grouping": "Connected components sharing reporting Account or exact normalized source/preprocessed text",
        "params": PARAMS,
    }
    write_json("label_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return train


def metrics(y, probs):
    pred = (probs > 0.5).astype(int)
    result = {
        "n": len(y), "subset_accuracy": float(accuracy_score(y, pred)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "per_label": {},
    }
    for k, label in enumerate(LABELS):
        result["per_label"][label] = {
            "support": int(y[:, k].sum()),
            "f1": float(f1_score(y[:, k], pred[:, k], zero_division=0)),
            "precision": float(precision_score(y[:, k], pred[:, k], zero_division=0)),
            "recall": float(recall_score(y[:, k], pred[:, k], zero_division=0)),
            "roc_auc": float(roc_auc_score(y[:, k], probs[:, k])) if len(np.unique(y[:, k])) == 2 else None,
        }
    return result


def load_runtime():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
    torch.set_num_threads(6)
    assert torch.cuda.is_available(), "This scheduled rebuild requires the available CUDA GPU"
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, token=False)
    return torch, AutoModelForSequenceClassification, tokenizer, get_linear_schedule_with_warmup, set_seed


def batches(df, tokenizer, torch, shuffle, seed):
    order = np.random.default_rng(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    for start in range(0, len(order), PARAMS["batch_size"]):
        rows = df.iloc[order[start:start + PARAMS["batch_size"]]]
        enc = tokenizer(rows.text.tolist(), truncation=True, max_length=PARAMS["max_length"], padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        if set(LABELS).issubset(rows.columns):
            enc["labels"] = torch.tensor(rows[LABELS].to_numpy(dtype=np.float32), device="cuda")
        yield enc


def fit(df, runtime, seed):
    torch, Model, tokenizer, scheduler_factory, set_seed = runtime
    set_seed(seed)
    model = Model.from_pretrained(
        MODEL_ID, num_labels=4, problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={v: k for k, v in enumerate(LABELS)},
        trust_remote_code=False, token=False,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS["learning_rate"], weight_decay=0)
    steps = ((len(df) + PARAMS["batch_size"] - 1) // PARAMS["batch_size"]) * PARAMS["epochs"]
    scheduler = scheduler_factory(optimizer, num_warmup_steps=0, num_training_steps=steps)
    losses = []
    for epoch in range(PARAMS["epochs"]):
        model.train()
        total = 0.0
        count = 0
        started = time.time()
        for enc in batches(df, tokenizer, torch, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss
            assert torch.isfinite(loss).item(), "Nonfinite training loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
            count += 1
        losses.append(total / count)
        print(f"epoch={epoch+1} loss={losses[-1]:.5f} seconds={time.time()-started:.1f}", flush=True)
    return model, losses


def predict(df, model, runtime):
    torch, _, tokenizer, _, _ = runtime
    model.eval()
    result = []
    with torch.inference_mode():
        for enc in batches(df, tokenizer, torch, shuffle=False, seed=SEED):
            enc.pop("labels", None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            result.append(torch.sigmoid(logits.float()).cpu().numpy())
    probs = np.concatenate(result) if result else np.empty((0, 4))
    assert probs.shape == (len(df), 4) and np.isfinite(probs).all()
    return probs


def train_all():
    df = audit()
    runtime = load_runtime()
    torch, _, tokenizer, _, _ = runtime
    write_json("environment.json", {
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0), "seed": SEED, "base_model": MODEL_ID,
        "precision": "bf16 autocast with float32 parameters", "params": PARAMS,
        "checkpoint_resume": False, "network_text_upload": False,
    })
    oof = np.zeros((len(df), 4))
    fold_results = []
    for fold in range(5):
        fold_dir = ROOT / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        val = df.loc[df.fold == fold]
        train = df.loc[df.fold != fold]
        print(f"FOLD {fold}: train={len(train)} validation={len(val)}", flush=True)
        model, losses = fit(train, runtime, SEED + fold)
        probs = predict(val, model, runtime)
        oof[val.index] = probs
        score = metrics(val[LABELS].values, probs)
        score.update({"fold": fold, "training_loss": losses})
        fold_results.append(score)
        np.save(fold_dir / "validation_probabilities.npy", probs)
        write_json(f"fold_{fold}/metrics.json", score)
        print(json.dumps(score, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    scores = metrics(df[LABELS].values, oof)
    scores["folds"] = fold_results
    scores["mean_fold_micro_f1"] = float(np.mean([x["micro_f1"] for x in fold_results]))
    scores["evaluation_status"] = "Independent grouped cross-validation, not a future-year external validation"
    write_json("cross_validation_metrics.json", scores)
    exported = df[["source_row", "Account number", "City", "fold", "group"] + LABELS].copy()
    for k, label in enumerate(LABELS):
        exported[f"prob_{label}"] = oof[:, k]
        exported[f"pred_{label}"] = (oof[:, k] > 0.5).astype(int)
    exported.to_csv(ROOT / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    print("FINAL TRAINING: all eligible unique human-labelled texts", flush=True)
    model, losses = fit(df, runtime, SEED + 100)
    destination = ROOT / "final_model"
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}
    write_json("model_manifest.json", {
        "files_sha256": hashes, "training_rows": len(df), "training_loss": losses,
        "base_model": MODEL_ID, "base_revision": model.config._commit_hash,
        "labels": LABELS, "threshold": "sigmoid(logit) > 0.5", "params": PARAMS,
        "human_source_sha256": hashlib.sha256((ROOT / "human_labels.csv").read_bytes()).hexdigest(),
        "limitations": "No new-year human validation; probability confidence is uncalibrated; inherited labels are not independently re-adjudicated.",
    })
    print("COMPLETE: final_model and audit outputs saved", flush=True)


def infer(path):
    runtime = load_runtime()
    torch, Model, _, _, _ = runtime
    source = Path(path)
    df = pd.read_csv(source, keep_default_na=False)
    source_rows = len(df)
    if "selected" in df:
        df = df.loc[df["selected"].astype(str).str.lower().eq("true")].reset_index(drop=True)
    assert {"Action", "Description"}.issubset(df.columns)
    df["text"] = [text_join(a, b, t) for a, b, t in zip(df.Action, df.Description, df.get("Title", pd.Series([""] * len(df))))]
    df["valid_text"] = df.text.map(normalize).str.len() > 0
    df["encoding_warning"] = df["text"].str.contains("\ufffd|ï¿½|â€™|â€“|Ã", regex=True)
    model = Model.from_pretrained(ROOT / "final_model", local_files_only=True, trust_remote_code=False).cuda()
    probs = np.full((len(df), 4), np.nan)
    probs[df.valid_text.values] = predict(df.loc[df.valid_text], model, runtime)
    for k, label in enumerate(LABELS):
        df[f"prob_{label}"] = probs[:, k]
        df[f"pred_{label}"] = pd.array(np.where(df.valid_text, (probs[:, k] > .5).astype(float), np.nan), dtype="Int64")
    # shortcut: 0.1 probability margin is a triage flag, not calibrated uncertainty.
    df["low_confidence_review"] = np.any(np.abs(probs - .5) < .1, axis=1) | ~df.valid_text.values
    df["no_type_predicted"] = np.nansum(probs > .5, axis=1) == 0
    df["human_review_status"] = "not_reviewed_for_this_model"
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    df["model_sha256"] = manifest["files_sha256"]["model.safetensors"]
    output = ROOT / (source.stem + "_classified.csv")
    df.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(source.stem + "_inference_summary.json", {
        "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": source_rows, "rows": len(df), "empty_text_rows": int((~df.valid_text).sum()),
        "selection": "Selected heat AP/all MP only; geographic-QC flags retained, no mapping silently corrected",
        "encoding_warning_rows": int(df.encoding_warning.sum()),
        "low_confidence_rows": int(df.low_confidence_review.sum()),
        "no_type_predicted": int(df.no_type_predicted.sum()),
        "output": output.name,
    })
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "train", "infer"])
    parser.add_argument("--input")
    args = parser.parse_args()
    assert normalize(" A, B! ") == "a b"
    assert text_join("", "", "title") == "title"
    if args.mode == "audit":
        audit()
    elif args.mode == "train":
        train_all()
    else:
        assert args.input, "--input is required"
        infer(args.input)
