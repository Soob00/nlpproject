"""
Unified inference script for RumourEval stance classification.

Usage:
    python inference/run_eval.py --model-size 1.5b --variant adv --split dev
    python inference/run_eval.py --model-size 3b   --variant zs  --split test
    python inference/run_eval.py --model-size 0.5b --variant ft  --split all

Arguments:
  --model-size  : 0.5b | 1.5b | 3b | 7b
  --variant     : zs (zero-shot base) | ft (fine-tuned) | adv (adversarial fine-tuned)
  --split       : dev | test | all  (all = dev + test)
  --device      : auto (default) | cpu | cuda
  --dtype       : auto (default) | float16 | float32 | bfloat16

Output:
    data/results/experiment_results_{SIZE}_{variant}.json
    Format: {"dev": {"c0": {golds, preds, n, macro_f1, per_class_f1}, ...}, "test": {...}}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference._common import (
    CONDITIONS, COND_TO_FIELD, INT2LABEL, SIZE_TAG,
    evaluate, load_dataset, load_model, resolve_device, resolve_dtype,
    result_path, run_condition, save_results,
)


def run_split(split: str, tokenizer, model) -> dict:
    print(f"\n{'='*60}")
    print(f"  Split: {split}")
    print(f"{'='*60}")
    dataset = load_dataset(split)
    print(f"  {len(dataset)} samples\n")

    cond_results: dict = {}
    t0 = time.time()

    for cond in CONDITIONS:
        field = COND_TO_FIELD[cond]
        print(f"\n--- {cond} ({field}) ---")
        raw     = run_condition(cond, dataset, tokenizer, model)
        metrics = evaluate(raw)
        cond_results[cond] = {
            "golds":         raw["golds"],
            "preds":         raw["preds"],
            "reply_ids":     raw["reply_ids"],
            "pred_labels":   raw["pred_labels"],
            "invalid_ids":   raw["invalid_ids"],
            "skipped_ids":   raw["skipped_ids"],
            "invalid_count": raw["invalid_count"],
            "skipped_count": raw["skipped_count"],
            "n":             raw["n"],
            "macro_f1":      metrics["macro_f1"],
            "per_class_f1":  metrics["per_class_f1"],
        }
        pc = metrics["per_class_f1"]
        print(f"  n={raw['n']} | macro={metrics['macro_f1']:.4f} | "
              f"sup={pc[0]:.3f} den={pc[1]:.3f} qry={pc[2]:.3f} com={pc[3]:.3f}")

    elapsed = time.time() - t0
    print(f"\n  Split done in {elapsed/60:.1f}m")
    return cond_results


def print_summary(size: str, variant: str, cond_results: dict, split: str) -> None:
    print(f"\n{'='*65}")
    print(f"  Results: {SIZE_TAG[size]}_{variant}  [{split}]")
    print(f"{'='*65}")
    print(f"  {'cond':<6} {'field':<12} {'macro':>7} {'sup':>6} {'den':>6} {'qry':>6} {'com':>7} {'n':>6}")
    print(f"  {'-'*58}")
    for cond in CONDITIONS:
        if cond not in cond_results:
            continue
        r  = cond_results[cond]
        pc = r["per_class_f1"]
        print(f"  {cond:<6} {COND_TO_FIELD[cond]:<12} {r['macro_f1']:>7.4f} "
              f"{pc[0]:>6.3f} {pc[1]:>6.3f} {pc[2]:>6.3f} {pc[3]:>7.3f} {r['n']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RumourEval stance inference")
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
    out    = result_path(args.model_size, args.variant)

    print(f"Model : {SIZE_TAG[args.model_size]}_{args.variant}")
    print(f"Device: {device}  dtype={dtype}")
    print(f"Splits: {splits}")
    print(f"Output: {out}\n")

    tokenizer, model = load_model(args.model_size, args.variant, dtype=dtype, device=device)

    for split in splits:
        cond_results = run_split(split, tokenizer, model)
        print_summary(args.model_size, args.variant, cond_results, split)
        save_results(out, split, cond_results)


if __name__ == "__main__":
    main()
