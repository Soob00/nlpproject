"""
RumourEval stance classification — LoRA fine-tuning script.

Usage:
    python training/train.py --model-size 1.5b --variant ft
    python training/train.py --model-size 3b   --variant adv
    python training/train.py --model-size 7b   --variant ft  --epochs 2 --lr 1e-4
    python training/train.py --model-size 0.5b --variant ft --device cpu --dtype float32

Arguments:
  --model-size : 0.5b | 1.5b | 3b | 7b
  --variant    : ft  (fine-tune on useful context only)
               | adv (adversarial: useful + conflicting + mixed)
  --epochs     : number of training epochs (default: 3)
  --lr         : learning rate (default: 2e-4)
  --lora-r     : LoRA rank (default: 8)
  --device     : auto | cuda | cpu
  --dtype      : auto | float16 | float32 | bfloat16
  --resume     : path to checkpoint to resume from (optional)

Output:
    models/qwen_{SIZE}_{variant}/          per-epoch checkpoints
    models/qwen_{SIZE}_{variant}/final/    final LoRA adapter
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    TRAINING_IMPORT_ERROR = exc
else:
    TRAINING_IMPORT_ERROR = None

ROOT      = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "context_conditions.json"
MODELS    = ROOT / "models"

BASE_MODELS: dict[str, str] = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

BATCH_SIZE: dict[str, int] = {
    "0.5b": 8,
    "1.5b": 2,
    "3b":   2,
    "7b":   1,
}

CPU_BATCH_SIZE: dict[str, int] = {
    "0.5b": 1,
    "1.5b": 1,
    "3b":   1,
    "7b":   1,
}

GRAD_ACCUM: dict[str, int] = {
    "0.5b": 2,
    "1.5b": 8,
    "3b":   8,
    "7b":   16,
}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float32 if device == "cpu" else torch.float16
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]

SYSTEM_PROMPT = (
    "You are a stance classification expert for social media discussions about rumours.\n\n"
    "Classify the stance of the TARGET reply using exactly one of these four labels:\n\n"
    "- support: The reply explicitly states the rumour IS true or confirmed.\n"
    "- deny: The reply explicitly states the rumour IS false or fabricated.\n"
    "- query: The reply asks for sources, evidence, or verification.\n"
    "- comment: Everything else. The reply does not directly address whether the rumour is true or false.\n\n"
    "Respond with ONLY one word: support, deny, query, or comment. No explanation."
)


def clean(text: str) -> str:
    return (
        text
        .replace("[Source]",     "Rumour post:")
        .replace("[Context]",    "Previous reply:")
        .replace("[Misleading]", "Another reply:")
        .replace("[Target]",     "Reply to classify:")
    )


def format_sample(d: dict, field: str) -> dict:
    user_content = (
        "Read the following and classify the stance of the 'Reply to classify'.\n\n"
        f"{clean(d[field])}\n\n"
        "Stance label (support / deny / query / comment):"
    )
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{d['label']}<|im_end|>"
        )
    }


def build_dataset(variant: str) -> Dataset:
    with open(DATA_FILE, encoding="utf-8") as f:
        full = json.load(f)

    train = [d for d in full if d["split"] == "train"]

    samples: list[dict] = []
    if variant == "ft":
        # useful context only
        for d in train:
            if d.get("useful") is not None:
                samples.append(format_sample(d, "useful"))
    elif variant == "adv":
        # adversarial: useful + conflicting + mixed
        for d in train:
            if d.get("useful") is not None:
                samples.append(format_sample(d, "useful"))
            if d.get("conflicting") is not None:
                samples.append(format_sample(d, "conflicting"))
            if d.get("mixed") is not None:
                samples.append(format_sample(d, "mixed"))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(f"Training samples ({variant}): {len(samples)}")
    return Dataset.from_list(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=["0.5b", "1.5b", "3b", "7b"])
    parser.add_argument("--variant",    required=True, choices=["ft", "adv"])
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-r",     type=int,   default=8)
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype",      default="auto",
                        choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Checkpoint path to resume from")
    args = parser.parse_args()

    if TRAINING_IMPORT_ERROR is not None:
        raise ImportError(
            "Training dependencies are missing. Install them with `pip install -r requirements.txt`."
        ) from TRAINING_IMPORT_ERROR

    size    = args.model_size
    variant = args.variant
    device  = resolve_device(args.device)
    dtype   = resolve_dtype(args.dtype, device)
    tag     = f"qwen_{size}_{variant}"
    out_dir = MODELS / tag
    is_cpu  = device == "cpu"

    print(f"\n{'='*60}")
    print(f"  Model : {BASE_MODELS[size]}")
    print(f"  Variant: {variant}  |  Tag: {tag}")
    print(f"  Device: {device}  |  dtype: {dtype}")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  LoRA-r: {args.lora_r}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")
    if is_cpu:
        print("[warning] CPU fine-tuning is supported for completeness, but it is very slow. "
              "Prefer CPU for evaluation/analysis and GPU for LoRA training when possible.\n")

    # data
    dataset = build_dataset(variant)

    # tokenizer
    base_model_name = BASE_MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    device_map = "auto" if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if is_cpu:
        model.to("cpu")
    model.config.use_cache = False

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # trainer config
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=CPU_BATCH_SIZE[size] if is_cpu else BATCH_SIZE[size],
        gradient_accumulation_steps=GRAD_ACCUM[size],
        learning_rate=args.lr,
        fp16=(device == "cuda" and dtype == torch.float16),
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        logging_steps=50,
        save_strategy="epoch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # save final LoRA adapter
    final_path = out_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved → {final_path}")


if __name__ == "__main__":
    main()
