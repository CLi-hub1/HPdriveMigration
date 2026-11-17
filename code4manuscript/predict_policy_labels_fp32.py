"""One FP32 prediction per exact text, shared across all policy measurement variants."""
from pathlib import Path
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

from config import POLICY_WORK
HERE = POLICY_WORK / "policy_inputs"
D = HERE / "data"
import rebuild_classifier as frozen
stems = ["policy_action_groups_2017_2023", "policy_action_groups_2017_2023_title_only_sensitivity", "full2019_questionnaire_sensitivity_groups"]
outputs = ["policy_action_groups_2017_2023_classified", "policy_action_groups_2017_2023_title_only_sensitivity_classified", "full2019_questionnaire_sensitivity_classified"]
frames, source_hashes = [], []
for stem in stems:
    path = D/(stem+".csv")
    source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    df = pd.read_csv(path,keep_default_na=False)
    df = df[df.selected.astype(str).str.lower().eq("true")].reset_index(drop=True)
    df["text"] = [frozen.text_join(a,b,t) for a,b,t in zip(df.Action,df.Description,df.Title)]
    assert df.text.map(frozen.normalize).str.len().gt(0).all()
    frames.append(df)
texts = sorted(set().union(*(set(df.text) for df in frames)))
runtime = frozen.load_runtime()
torch, Model, tokenizer = runtime[:3]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
model = Model.from_pretrained(frozen.ROOT/"final_model",local_files_only=True,trust_remote_code=False).float().cuda().eval()
probabilities = []
with torch.inference_mode():
    for start in range(0,len(texts),8):
        enc = tokenizer(texts[start:start+8],max_length=512,padding="max_length",truncation=True,return_tensors="pt")
        p = torch.sigmoid(model(**{k:v.cuda() for k,v in enc.items()}).logits).cpu().numpy()
        probabilities.append(p)
        if start%800==0:
            print(f"FP32 unique texts {start}/{len(texts)}",flush=True)
probs = np.concatenate(probabilities)
assert probs.shape == (len(texts),4) and np.isfinite(probs).all()
cache = pd.DataFrame({"text":texts})
for k,label in enumerate(frozen.LABELS):
    cache["prob_"+label] = probs[:,k]
    cache["pred_"+label] = (probs[:,k]>.5).astype(int)
cache.to_csv(D/"shared_exact_text_FP32_predictions.csv",index=False)
cache_sha = hashlib.sha256((D/"shared_exact_text_FP32_predictions.csv").read_bytes()).hexdigest()
model_sha = hashlib.sha256((frozen.ROOT/"final_model/model.safetensors").read_bytes()).hexdigest()
lookup = cache.set_index("text")
old_folder = HERE/"bf16_inference_audit"
old_folder.mkdir(exist_ok=True)
for stem,out_stem,df,input_sha in zip(stems,outputs,frames,source_hashes):
    output = D/(out_stem+".csv")
    metadata_path = output.with_suffix(".inference.json")
    if output.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "bf16" in previous.get("inference_precision", "").lower():
            shutil.copy2(output,old_folder/output.name)
            shutil.copy2(metadata_path,old_folder/metadata_path.name)
    assert hashlib.sha256((D/(stem+".csv")).read_bytes()).hexdigest()==input_sha
    df = df.drop(columns=[c for c in df if c.startswith(("prob_","pred_"))],errors="ignore")
    df = df.join(lookup,on="text",validate="many_to_one")
    assert df[["prob_"+c for c in frozen.LABELS]].notna().all().all()
    df["valid_text"] = True
    df["low_confidence_review"] = df[["prob_"+c for c in frozen.LABELS]].sub(.5).abs().lt(.1).any(axis=1)
    df["no_type_predicted"] = df[["pred_"+c for c in frozen.LABELS]].sum(axis=1).eq(0)
    df["human_review_status"],df["model_sha256"] = "not_reviewed_for_this_model",model_sha
    df.to_csv(output,index=False,encoding="utf-8-sig")
    metadata = {"input":stem+".csv","input_sha256":input_sha,
        "output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"rows":len(df),"model_sha256":model_sha,
        "shared_cache_sha256":cache_sha,"inference_precision":"FP32; TF32 disabled; deterministic algorithms; fixed padding 512",
        "one_prediction_per_exact_text":True,"threshold":"sigmoid(logit) > 0.5","retrained":False,
        "torch":torch.__version__,"transformers":__import__("transformers").__version__}
    output.with_suffix(".inference.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(out_stem,len(df),flush=True)
changed=[]
comparison_variants = 0
for out_stem in outputs:
    old_path = old_folder / (out_stem + ".csv")
    if not old_path.exists():
        continue
    comparison_variants += 1
    old=pd.read_csv(old_path,keep_default_na=False).drop_duplicates("text")
    merged=cache.merge(old[["text"]+["pred_"+c for c in frozen.LABELS]],on="text",suffixes=("_FP32","_BF16"))
    for label in frozen.LABELS:
        q=merged[merged["pred_"+label+"_FP32"]!=merged["pred_"+label+"_BF16"]].copy()
        for _,r in q.iterrows():
            changed.append(dict(variant=out_stem,text=r.text,label=label,FP32_probability=r["prob_"+label],FP32_label=r["pred_"+label+"_FP32"],BF16_label=r["pred_"+label+"_BF16"]))
pd.DataFrame(changed).to_csv(D/"FP32_vs_BF16_threshold_audit.csv",index=False)
(D/"shared_inference_manifest.json").write_text(json.dumps({"unique_exact_texts":len(texts),"cache_sha256":cache_sha,
    "model_sha256":model_sha,"source_sha256":dict(zip(stems,source_hashes)),"precision":"FP32","max_length":512,
    "BF16_variants_compared":comparison_variants,
    "threshold_changes_vs_previous_BF16_variant_runs":len(changed) if comparison_variants else None,
    "new_human_validation":False},indent=2),encoding="utf-8")
print("SHARED EXACT-TEXT INFERENCE COMPLETE",flush=True)
