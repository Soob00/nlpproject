# RumourEval Context Sensitivity Experiments

This repository builds and evaluates Qwen2.5 stance classifiers on RumourEval 2019.
The main question is how different context conditions affect stance prediction and
model confidence for social media rumour replies.

## Pipeline

1. Build the processed dataset:

```bash
python build_dataset.py
```

This writes `data/processed/context_conditions.json` with six conditions:

| key | field | meaning |
| --- | --- | --- |
| `c0` | `reply_only` | target reply only |
| `c1` | `useful` | source post plus parent context when available |
| `c2` | `irrelevant` | source post plus unrelated comment context |
| `c3` | `conflicting` | source post plus opposing stance context |
| `c4` | `mixed` | useful context plus misleading context |
| `c5` | `lexical` | label-biased lexical distractor |

Invalid c2/c3/c4 examples are stored as `null` and skipped for that condition.

2. Train LoRA adapters:

```bash
python training/train.py --model-size 1.5b --variant ft
python training/train.py --model-size 1.5b --variant adv
```

`ft` trains on useful context only. `adv` trains on useful, conflicting, and mixed
conditions. Final adapters are saved under `models/qwen_{size}_{variant}/final/`.

3. Run evaluation:

```bash
python inference/run_eval.py --model-size 1.5b --variant adv --split dev
python inference/run_eval.py --model-size 3b --variant zs --split all
```

CPU-only evaluation:

```bash
python inference/run_eval_cpu.py --model-size 0.5b --variant zs --split dev
python inference/run_eval_cpu.py --model-size 1.5b --variant adv --split dev
```

GPU evaluation uses the same script with `--device cuda`:

```bash
python inference/run_eval_gpu.py --model-size 3b --variant zs --split dev
```

Outputs are written to `data/results/experiment_results_{SIZE}_{variant}.json`.
Each condition stores `golds`, `preds`, `reply_ids`, invalid/skipped IDs, macro-F1,
and per-class F1.

4. Extract confidence scores:

```bash
python inference/run_conf.py --model-size 1.5b --variant ft --split dev
python inference/run_conf.py --model-size 1.5b --variant adv --split dev
python inference/run_conf_cpu.py --model-size 0.5b --variant zs --split dev
python inference/run_conf_gpu.py --model-size 3b --variant zs --split dev
python inference/run_conf_gpu.py --model-size 1.5b --variant adv --split dev
```

Outputs are written to `data/results/confidence_results_{SIZE}_{variant}.json`.

5. Generate analysis tables:

```bash
python analysis/run_all.py
```

CSV outputs go to `analysis/output/`; figures go to `figures/`.

## CPU vs GPU

- Analysis scripts are CPU-only.
- Inference and confidence extraction can run on CPU with `--device cpu --dtype float32`.
- GPU inference/training mainly differs in device placement and dtype, usually `float16`.
- LoRA fine-tuning on CPU is technically supported through `training/train_cpu.py`, but it is very slow. Use GPU for full training when available.
- `peft` is required only for `ft` and `adv` LoRA adapters, not for zero-shot CPU evaluation.

## Notes

- The default analysis split is set in `analysis/_paths.py` as `EVAL_SPLIT`.
- Result filenames use `zs`, `ft`, and `adv` consistently.
- Invalid model generations are kept in the result arrays as prediction `-1`
  (`invalid`) so they are counted as errors instead of silently removed.
