import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, CONF_FILES, CONDITIONS, LABELS, EVAL_SPLIT, load_conf

label_order = LABELS

for fname, model in [(DATA_RESULTS / k, v) for k, v in CONF_FILES.items()]:
    if not fname.exists():
        print(f"[skip] {fname.name} not found")
        continue
    data = load_conf(fname, split=EVAL_SPLIT)
    if not data:
        print(f"[skip] no records for split='{EVAL_SPLIT}' in {fname.name}")
        continue
    print(f'\n========== {model} ==========')
    for cond in CONDITIONS:
        print(f'\n  [{cond}]')
        buckets = {lbl: {'correct': [], 'wrong': []} for lbl in label_order}
        for row in data:
            lbl = row['true_label']
            if lbl not in buckets:
                continue
            conf = row[f'{cond}_conf']
            correct = row[f'{cond}_correct']
            if correct:
                buckets[lbl]['correct'].append(conf)
            else:
                buckets[lbl]['wrong'].append(conf)

        print(f'  {"class":8s}  {"correct":>8s}(n)   {"wrong":>8s}(n)   {"diff(W-C)":>9s}')
        for lbl in label_order:
            c = buckets[lbl]['correct']
            w = buckets[lbl]['wrong']
            c_mean = sum(c)/len(c) if c else float('nan')
            w_mean = sum(w)/len(w) if w else float('nan')
            diff = w_mean - c_mean if c and w else float('nan')
            c_str = f'{c_mean:.4f}' if c else '   nan'
            w_str = f'{w_mean:.4f}' if w else '   nan'
            d_str = f'{diff:+.4f}' if (c and w) else '    nan'
            print(f'  {lbl:8s}  {c_str:>8s}({len(c):4d})   {w_str:>8s}({len(w):4d})   {d_str:>9s}')
