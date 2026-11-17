"""Equal-budget grouped CV comparison; never reuse fine-tuned fold weights."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time

from config import CLASSIFIER_WORK, OUTPUT_ROOT
ML = CLASSIFIER_WORK
OUT = OUTPUT_ROOT / "classifier_search"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true"})
spec = importlib.util.spec_from_file_location("rebuild", Path(__file__).resolve().parent / "rebuild_classifier.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

SEED = 20261020
N_TRIALS = 12
BASE_DIR = Path(os.environ.get("MANUSCRIPT_BASE_MODEL", ML / "base_model"))
# Explicit local path prohibits downloads or use of the all-label fine-tuned checkpoint.
base.MODEL_ID = str(BASE_DIR)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare():
    df = pd.read_csv(ML / "training_rows.csv", keep_default_na=False)
    assert len(df) == 964 and df.group.nunique() == 95 and df.text_key.is_unique
    assert set(df.fold.unique()) == set(range(5))
    splits = []
    for fold in range(5):
        tr = df.loc[df.fold != fold]
        va = df.loc[df.fold == fold]
        for key in ["group", "Account number", "text_key", "merge_key"]:
            assert not (set(tr[key]) - {""}) & (set(va[key]) - {""}), key
        splits.append({"fold": fold, "train_n": len(tr), "validation_n": len(va),
                       "train_groups": tr.group.nunique(), "validation_groups": va.group.nunique()})
    grid = [{"learning_rate": lr, "epochs": ep, "batch_size": bs, "max_length": 512}
            for lr, ep, bs in itertools.product([1e-5, 3e-5, 5e-5], [3, 5], [8, 16])]
    rng = np.random.default_rng(SEED)
    random = [{"learning_rate": float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-5)))),
               "epochs": int(rng.choice([3, 5])), "batch_size": int(rng.choice([8, 16])),
               "max_length": 512} for _ in range(N_TRIALS)]
    manifest = {
        "status": "DESIGN FROZEN BEFORE SEARCH",
        "training_csv_sha256": sha(ML / "training_rows.csv"),
        "human_labels_sha256": sha(ML / "human_labels.csv"),
        "base_revision": BASE_DIR.name,
        "base_weights_sha256": {p.name: sha(p) for p in BASE_DIR.iterdir()
                                if p.suffix in [".safetensors", ".bin"]},
        "script_sha256": sha(Path(__file__)),
        "rows": len(df), "connected_groups": int(df.group.nunique()), "folds": splits,
        "seed": SEED, "candidate_budget_per_strategy": N_TRIALS,
        "fit_budget_per_strategy": N_TRIALS * 5, "total_fit_budget": N_TRIALS * 5 * 3,
        "selection_objective": "Unweighted mean of five validation-fold micro F1 scores",
        "search_bounds": {"learning_rate": [1e-5, 5e-5], "epochs": [3, 5],
                          "batch_size": [8, 16], "max_length": 512},
        "grid_candidates": grid, "random_candidates": random,
        "random_algorithm": "NumPy seeded log-uniform learning rate, uniform discrete epochs/batch",
        "bayesian_algorithm": "Optuna TPESampler(seed, n_startup_trials=4), 12 sequential trials; no pruning",
        "shared_initialization": "Every candidate/fold loads fresh public pretrained weights; fresh head initialized with seed+fold. No fold/checkpoint resume.",
        "optimizer": "AdamW, weight_decay=0, linear LR decay, warmup=0, clip_grad_norm=1",
        "precision": "bf16 autocast; float32 model parameters",
        "threshold": "sigmoid(logit)>0.5; threshold not tuned",
        "metrics": {"Loss": "Mean binary cross-entropy over all labels and validation rows",
                    "Accuracy": "Subset accuracy: all four labels match",
                    "Precision": "micro", "Recall": "micro", "F1": "micro",
                    "ROC AUC": "Unweighted mean of four label ROC AUCs"},
        "evaluation_limit": "The displayed best-candidate CV folds also selected hyperparameters. Selection-optimistic internal CV, not nested or external validation.",
        "nested_cv_minimum": "5 outer grouped folds x 3 inner grouped folds x 12 candidates x 3 strategies = 540 inner fits +15 outer refits, each initialized anew. New-year independently labelled holdout still required.",
        "existing_regression_classifier_replaced": False,
        "network_upload": False, "paid_api": False,
        "result_cache": "Only exact completed candidate/fold results from this immutable design may resume; never model weights.",
    }
    path = OUT / "search_design_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert existing == manifest, "Design changed; use a different output directory."
    else:
        write_json(path, manifest)
    return df, manifest


def fold_scores(y, probs):
    probs = np.clip(probs.astype(float), 1e-7, 1 - 1e-7)
    preds = (probs > .5).astype(int)
    return {"Loss": float(-(y * np.log(probs) + (1-y) * np.log(1-probs)).mean()),
            "Accuracy": float(accuracy_score(y, preds)),
            "Precision": float(precision_score(y, preds, average="micro", zero_division=0)),
            "Recall": float(recall_score(y, preds, average="micro", zero_division=0)),
            "F1": float(f1_score(y, preds, average="micro", zero_division=0)),
            "ROC AUC": float(roc_auc_score(y, probs, average="macro"))}


def evaluate(strategy, trial, params, df, runtime):
    base.PARAMS = dict(params)
    directory = OUT / strategy / f"trial_{trial:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "parameters.json", params)
    fold_results = []
    for fold in range(5):
        complete = directory / f"fold_{fold}.json"
        probabilities = directory / f"fold_{fold}_probabilities.npy"
        if complete.exists() and probabilities.exists():
            row = json.loads(complete.read_text(encoding="utf-8"))
            assert row["parameters"] == params and row["seed"] == SEED + fold
            fold_results.append(row)
            print(f"CACHED {strategy} trial={trial} fold={fold}", flush=True)
            continue
        started = time.time()
        train, val = df.loc[df.fold != fold], df.loc[df.fold == fold]
        print(f"FIT {strategy} trial={trial} fold={fold} params={params}", flush=True)
        model, losses = base.fit(train, runtime, SEED + fold)
        probs = base.predict(val, model, runtime)
        np.save(probabilities, probs)
        row = {"strategy": strategy, "trial": trial, "fold": fold, "seed": SEED + fold,
               "parameters": params, "n_train": len(train), "n_validation": len(val),
               "training_losses": losses, "seconds": time.time()-started,
               **fold_scores(val[base.LABELS].to_numpy(), probs)}
        write_json(complete, row)
        fold_results.append(row)
        del model
        runtime[0].cuda.empty_cache()
        print(f"DONE {strategy} trial={trial} fold={fold} F1={row['F1']:.6f} seconds={row['seconds']:.1f}", flush=True)
    score = float(np.mean([r["F1"] for r in fold_results]))
    write_json(directory / "candidate_summary.json", {"strategy": strategy, "trial": trial,
               "parameters": params, "mean_fold_micro_f1": score, "folds": fold_results})
    return score


def summarize(df):
    candidates, folds, selected, oof_rows = [], [], [], []
    for strategy in ["Grid search", "Random search", "Bayesian optimization"]:
        records = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted((OUT / strategy).glob("trial_*/candidate_summary.json"))]
        assert len(records) == N_TRIALS
        # Stable tie break: first evaluated candidate wins.
        best = max(records, key=lambda x: x["mean_fold_micro_f1"])
        for row in records:
            candidates.append({k: v for k, v in row.items() if k not in ["folds", "parameters"]}
                              | row["parameters"] | {"selected": row["trial"] == best["trial"]})
            for fold in row["folds"]:
                folds.append({k: v for k, v in fold.items() if k not in ["training_losses", "parameters"]}
                             | row["parameters"] | {"selected": row["trial"] == best["trial"]})
        for fold in best["folds"]:
            selected.append({k: v for k, v in fold.items() if k not in ["training_losses", "parameters"]}
                            | best["parameters"])
            val = df.loc[df.fold == fold["fold"]]
            probs = np.load(OUT / strategy / f"trial_{best['trial']:02d}" / f"fold_{fold['fold']}_probabilities.npy")
            for pos, (_, r) in enumerate(val.iterrows()):
                result = {"strategy": strategy, "trial": best["trial"], "fold": fold["fold"],
                          "source_row": r.source_row, "group": r.group}
                for j, label in enumerate(base.LABELS):
                    result[label] = int(r[label])
                    result[f"prob_{label}"] = float(probs[pos, j])
                oof_rows.append(result)
    for name, values in [("all_candidates.csv", candidates), ("all_folds.csv", folds),
                         ("selected_candidate_fold_metrics.csv", selected),
                         ("selected_candidate_oof_predictions.csv", oof_rows)]:
        pd.DataFrame(values).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write_json(OUT / "completion.json", {"status": "COMPLETE", "completed_candidate_count": len(candidates),
               "completed_fold_count": len(folds), "selected_fold_count": len(selected),
               "design_sha256": sha(OUT / "search_design_manifest.json"),
               "all_folds_sha256": sha(OUT / "all_folds.csv"),
               "interpretation": "Selected internal grouped CV; not an unbiased post-tuning external benchmark.",
               "frozen_regression_classifier_changed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    df, manifest = prepare()
    if not args.run:
        print("Design saved; pass --run to execute.")
        sys.exit(0)
    runtime = base.load_runtime()
    import transformers
    write_json(OUT / "runtime.json", {"executable": sys.executable, "torch": runtime[0].__version__,
               "transformers": transformers.__version__, "optuna": optuna.__version__,
               "gpu": runtime[0].cuda.get_device_name(0), "free_total_GPU_bytes": runtime[0].cuda.mem_get_info()})
    for strategy, key in [("Grid search", "grid_candidates"), ("Random search", "random_candidates")]:
        for trial, params in enumerate(manifest[key]):
            evaluate(strategy, trial, params, df, runtime)
    # A fresh seeded study deterministically replays cached complete candidate results
    # before requesting subsequent candidates. No validation metrics are fabricated.
    study = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4))
    def objective(trial):
        params = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                  "epochs": trial.suggest_categorical("epochs", [3, 5]),
                  "batch_size": trial.suggest_categorical("batch_size", [8, 16]), "max_length": 512}
        return evaluate("Bayesian optimization", trial.number, params, df, runtime)
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(OUT / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    summarize(df)
    print("COMPLETE: 36 candidate evaluations / 180 fresh-fold fits.", flush=True)
