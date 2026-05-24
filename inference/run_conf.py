"""
Confidence score extraction for calibration analysis.

Extracts per-sample, per-condition confidence as max softmax probability
over the four candidate label tokens {support, deny, query, comment}.

Usage:
    python inference/run_conf.py --model-size 1.5b --variant ft  --split dev
    python inference/run_conf.py --model-size 1.5b --variant adv --split all
    python inference/run_conf.py --model-size 0.5b --variant zs --device cpu

Output:
    data/results/confidence_results_{SIZE}_{variant}.json
    Format: list of records, one per reply, with fields:
      reply_id, split, true_label,
      {cond}_pred, {cond}_conf, {cond}_correct   for each cond in c0..c5
      (skipped conditions have null values)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference._common import (
    CONDITIONS, COND_TO_FIELD, LABEL_ORDER, SIZE_TAG,
    build_prompt, conf_path, load_dataset, load_model, model_device,
    resolve_device, resolve_dtype,
)


def get_label_token_ids(tokenizer) -> dict[str, int]:
    """
    Returns the single token ID for each stance label.
    Assumes each label tokenises to exactly one token (true for Qwen tokenisers).
    """
    ids: dict[str, int] = {}
    for lbl in LABEL_ORDER:
        toks = tokenizer.encode(lbl, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Label '{lbl}' encodes to {len(toks)} tokens, expected 1.")
        ids[lbl] = toks[0]
    return ids


def predict_with_conf(
    text: str,
    tokenizer,
    model,
    label_token_ids: dict[str, int],
) -> tuple[str, float, float, float]:
    """
    Returns (predicted_label, confidence, entropy, margin).
    confidence = max softmax prob over the 4 label tokens at the first generated position.
    """
    msgs  = build_prompt(text)
    inp   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids   = tokenizer(inp, return_tensors="pt").input_ids.to(model_device(model))

    with torch.no_grad():
        out = model(ids)
        logits_last = out.logits[0, -1, :]  # (vocab_size,)

    # extract logits for the 4 label tokens and softmax
    label_logits = torch.tensor(
        [logits_last[label_token_ids[lbl]].item() for lbl in LABEL_ORDER],
        dtype=torch.float32,
    )
    probs = F.softmax(label_logits, dim=0)
    best_idx  = int(probs.argmax())
    pred_lbl  = LABEL_ORDER[best_idx]
    confidence = float(probs[best_idx])
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    top2 = torch.topk(probs, k=2).values
    margin = float(top2[0] - top2[1])

    return pred_lbl, confidence, entropy, margin


def run_confidence(
    split: str,
    tokenizer,
    model,
    label_token_ids: dict[str, int],
    log_interval: int = 100,
) -> list[dict[str, Any]]:
    dataset = load_dataset(split)
    print(f"\n  {len(dataset)} samples in split='{split}'")

    records: list[dict[str, Any]] = []

    for i, d in enumerate(dataset):
        rec: dict[str, Any] = {
            "reply_id":   d["reply_id"],
            "split":      split,
            "true_label": d["label"],
        }
        for cond in CONDITIONS:
            field = COND_TO_FIELD[cond]
            text  = d.get(field)
            if text is None:
                rec[f"{cond}_pred"]    = None
                rec[f"{cond}_conf"]    = None
                rec[f"{cond}_entropy"] = None
                rec[f"{cond}_margin"]  = None
                rec[f"{cond}_correct"] = None
                continue
            pred, conf, entropy, margin = predict_with_conf(text, tokenizer, model, label_token_ids)
            rec[f"{cond}_pred"]    = pred
            rec[f"{cond}_conf"]    = round(conf, 6)
            rec[f"{cond}_entropy"] = round(entropy, 6)
            rec[f"{cond}_margin"]  = round(margin, 6)
            rec[f"{cond}_correct"] = int(pred == d["label"])

        records.append(rec)
        if (i + 1) % log_interval == 0:
            print(f"  {i+1}/{len(dataset)}")

    return records


def save_conf(out_path: Path, new_records: list[dict]) -> None:
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        # remove records for the same split before merging
        split_val = new_records[0]["split"] if new_records else None
        existing  = [r for r in existing if r.get("split") != split_val]
        combined  = existing + new_records
    else:
        combined = new_records

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(combined)} records total)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence score extraction")
    parser.add_argument("--model-size", required=True, choices=["0.5b", "1.5b", "3b", "7b"])
    parser.add_argument("--variant",    required=True, choices=["zs", "ft", "adv"])
    parser.add_argument("--split",      default="dev",  choices=["dev", "test", "all"])
    parser.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype",      default="auto",
                        choices=["auto", "float16", "float32", "bfloat16"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype  = resolve_dtype(args.dtype, device)
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    out    = conf_path(args.model_size, args.variant)

    print(f"Model : {SIZE_TAG[args.model_size]}_{args.variant}")
    print(f"Device: {device}  dtype={dtype}")
    print(f"Splits: {splits}")
    print(f"Output: {out}\n")

    tokenizer, model = load_model(args.model_size, args.variant, dtype=dtype, device=device)
    label_token_ids  = get_label_token_ids(tokenizer)
    print(f"Label token IDs: {label_token_ids}\n")

    for split in splits:
        print(f"\n{'='*50}\n  Split: {split}\n{'='*50}")
        records = run_confidence(split, tokenizer, model, label_token_ids)
        save_conf(out, records)


if __name__ == "__main__":
    main()
