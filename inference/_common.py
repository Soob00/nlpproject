"""
Shared constants and functions for all inference scripts.

Model naming convention:
  size    : 0.5b | 1.5b | 3b | 7b
  variant : zs (zero-shot) | ft (fine-tuned) | adv (adversarial fine-tuned)

Result file format  (data/results/experiment_results_{SIZE}_{variant}.json):
  {
    "dev":  {"c0": {"golds":[int], "preds":[int], "n":int, "macro_f1":float, "per_class_f1":[f,f,f,f]}, ...},
    "test": {"c0": {...}, ...}   # added when test data is available
  }

Label ints: 0=support  1=deny  2=query  3=comment
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_FILE  = ROOT / "data" / "processed" / "context_conditions.json"
RESULTS_DIR = ROOT / "data" / "results"

# ── condition mapping ──────────────────────────────────────────────────────────
CONDITIONS: list[str] = ["c0", "c1", "c2", "c3", "c4", "c5"]
COND_TO_FIELD: dict[str, str] = {
    "c0": "reply_only",
    "c1": "useful",
    "c2": "irrelevant",
    "c3": "conflicting",
    "c4": "mixed",
    "c5": "lexical",
}

# ── label mapping ──────────────────────────────────────────────────────────────
LABEL_ORDER: list[str]      = ["support", "deny", "query", "comment"]
LABEL_MAP:   dict[str, int] = {label: i for i, label in enumerate(LABEL_ORDER)}
INT2LABEL:   dict[int, str] = {v: k for k, v in LABEL_MAP.items()} | {-1: "invalid"}
VALID_LABELS: set[str]      = set(LABEL_MAP)
INVALID_LABEL_ID = -1

# ── model paths ────────────────────────────────────────────────────────────────
BASE_MODELS: dict[str, str] = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

MODEL_PATHS: dict[tuple[str, str], str] = {
    # zero-shot: base model from HuggingFace (no local checkpoint needed)
    ("0.5b", "zs"):  "Qwen/Qwen2.5-0.5B-Instruct",
    ("1.5b", "zs"):  "Qwen/Qwen2.5-1.5B-Instruct",
    ("3b",   "zs"):  "Qwen/Qwen2.5-3B-Instruct",
    ("7b",   "zs"):  "Qwen/Qwen2.5-7B-Instruct",
    # fine-tuned: local LoRA-merged checkpoint
    ("0.5b", "ft"):  str(ROOT / "models" / "qwen_0.5b_ft"  / "final"),
    ("1.5b", "ft"):  str(ROOT / "models" / "qwen_1.5b_ft"  / "final"),
    ("3b",   "ft"):  str(ROOT / "models" / "qwen_3b_ft"    / "final"),
    ("7b",   "ft"):  str(ROOT / "models" / "qwen_7b_ft"    / "final"),
    # adversarial fine-tuned: local LoRA-merged checkpoint
    ("0.5b", "adv"): str(ROOT / "models" / "qwen_0.5b_adv" / "final"),
    ("1.5b", "adv"): str(ROOT / "models" / "qwen_1.5b_adv" / "final"),
    ("3b",   "adv"): str(ROOT / "models" / "qwen_3b_adv"   / "final"),
    ("7b",   "adv"): str(ROOT / "models" / "qwen_7b_adv"   / "final"),
}

# ── output file naming ─────────────────────────────────────────────────────────
SIZE_TAG: dict[str, str] = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B"}

def result_path(size: str, variant: str) -> Path:
    return RESULTS_DIR / f"experiment_results_{SIZE_TAG[size]}_{variant}.json"

def conf_path(size: str, variant: str) -> Path:
    return RESULTS_DIR / f"confidence_results_{SIZE_TAG[size]}_{variant}.json"

# ── prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a stance classification expert for social media discussions about rumours.\n\n"
    "Classify the stance of the TARGET reply using exactly one of these four labels:\n\n"
    "- support: The reply explicitly states the rumour IS true or confirmed.\n"
    "- deny: The reply explicitly states the rumour IS false or fabricated.\n"
    "- query: The reply asks for sources, evidence, or verification.\n"
    "- comment: Everything else. When in doubt, choose comment.\n\n"
    "Respond with ONLY one word: support, deny, query, or comment. No explanation."
)


def build_prompt(text: str) -> list[dict[str, str]]:
    cleaned = (
        text
        .replace("[Source]",     "Rumour post:")
        .replace("[Context]",    "Previous reply:")
        .replace("[Misleading]", "Another reply:")
        .replace("[Target]",     "Reply to classify:")
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            "Read the following and classify the stance of the 'Reply to classify'.\n\n"
            f"{cleaned}\n\n"
            "Stance label (support / deny / query / comment):"
        )},
    ]

# ── model loading ──────────────────────────────────────────────────────────────

def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float32 if device == "cpu" else torch.float16
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model(
    size: str,
    variant: str,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    key = (size, variant)
    if key not in MODEL_PATHS:
        raise ValueError(f"No model path for ({size}, {variant}). Available: {list(MODEL_PATHS)}")
    path = MODEL_PATHS[key]
    resolved_device = resolve_device(device)
    device_map = "auto" if resolved_device == "cuda" else None

    if variant == "zs":
        print(f"Loading base model: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map=device_map)
    else:
        # LoRA adapter: load base model first, then merge adapter
        if PeftModel is None:
            raise ImportError("peft is required to load ft/adv LoRA adapters. Install requirements.txt first.")
        base_name = BASE_MODELS[size]
        print(f"Loading base model: {base_name}")
        print(f"Applying LoRA adapter: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base, path)

    if resolved_device == "cpu":
        model.to("cpu")
    model.eval()
    print(f"Loaded on {model_device(model)} with dtype={dtype}\n")
    return tokenizer, model

# ── inference ──────────────────────────────────────────────────────────────────

def parse_label(raw: str) -> str:
    cleaned = raw.strip().lower()
    tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in cleaned.split()]
    if tokens and tokens[0] in VALID_LABELS:
        return tokens[0]
    for lbl in LABEL_ORDER:
        if lbl in tokens:
            return lbl
    return "invalid"


def predict(text: str, tokenizer, model, max_new_tokens: int = 10) -> str:
    msgs  = build_prompt(text)
    inp   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids   = tokenizer(inp, return_tensors="pt").input_ids.to(model_device(model))
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return parse_label(raw)


def run_condition(
    cond: str,
    dataset: list[dict],
    tokenizer,
    model,
    log_interval: int = 100,
) -> dict[str, Any]:
    field = COND_TO_FIELD[cond]
    golds: list[int] = []
    preds: list[int] = []
    reply_ids: list[str] = []
    pred_labels: list[str] = []
    invalid_ids: list[str] = []
    skipped_ids: list[str] = []
    invalid_count = skipped_count = 0
    total = len(dataset)

    for i, d in enumerate(dataset):
        if d.get(field) is None:
            skipped_count += 1
            skipped_ids.append(d["reply_id"])
            continue
        p = predict(d[field], tokenizer, model)
        if p == "invalid":
            invalid_count += 1
            pred_id = INVALID_LABEL_ID
            invalid_ids.append(d["reply_id"])
        else:
            pred_id = LABEL_MAP[p]
        golds.append(LABEL_MAP[d["label"]])
        preds.append(pred_id)
        reply_ids.append(d["reply_id"])
        pred_labels.append(p)
        if (i + 1) % log_interval == 0:
            print(f"  [{cond}] {i+1}/{total} | valid={len(golds)} invalid={invalid_count}")

    return {
        "golds":         golds,
        "preds":         preds,
        "reply_ids":     reply_ids,
        "pred_labels":   pred_labels,
        "invalid_ids":   invalid_ids,
        "skipped_ids":   skipped_ids,
        "invalid_count": invalid_count,
        "skipped_count": skipped_count,
        "n":             len(golds),
    }


def evaluate(raw: dict[str, Any]) -> dict[str, Any]:
    golds, preds = raw["golds"], raw["preds"]
    if not golds:
        return {"macro_f1": 0.0, "per_class_f1": [0.0] * 4}
    macro     = float(f1_score(golds, preds, average="macro",  zero_division=0))
    per_class = f1_score(golds, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return {"macro_f1": round(macro, 4), "per_class_f1": [round(x, 4) for x in per_class.tolist()]}

# ── data loading ───────────────────────────────────────────────────────────────

def load_dataset(split: str) -> list[dict]:
    with open(DATA_FILE, encoding="utf-8") as f:
        full = json.load(f)
    data = [d for d in full if d.get("split") == split]
    if not data:
        raise ValueError(f"No records found for split='{split}' in {DATA_FILE}")
    return data

# ── result saving ──────────────────────────────────────────────────────────────

def save_results(
    out_path: Path,
    split: str,
    cond_results: dict[str, dict],
) -> None:
    """
    Merge new split results into existing file (creates file if absent).
    cond_results: {c0..c5: {golds, preds, n, macro_f1, per_class_f1}}
    """
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        # handle old format (no split wrapper) transparently
        if not any(k in existing for k in ("dev", "test")):
            existing = {"dev": existing}
    else:
        existing = {}

    existing[split] = cond_results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out_path}")
