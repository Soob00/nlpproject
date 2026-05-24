"""
Build context_conditions.json from raw RumourEval 2019 data.

Processes train, dev, and test splits.

Inputs (relative to project root):
  data/raw/train-key.json
  data/raw/dev-key.json
  data/rumoureval2019/final-eval-key.json
  data/raw/reddit-training-data/
  data/raw/reddit-dev-data/
  data/raw/twitter-english/
  data/rumoureval2019/test-data/rumoureval-2019-test-data/reddit-test-data/
  data/rumoureval2019/test-data/rumoureval-2019-test-data/twitter-en-test-data/

Output:
  data/processed/context_conditions.json
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── label key files ────────────────────────────────────────────────────────────
def _load_key(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("subtaskaenglish", {})

train_labels = _load_key(ROOT / "data/raw/train-key.json")
dev_labels   = _load_key(ROOT / "data/raw/dev-key.json")
test_labels  = _load_key(ROOT / "data/rumoureval2019/final-eval-key.json")
label_map    = {**train_labels, **dev_labels, **test_labels}

print(f"train: {len(train_labels)}  dev: {len(dev_labels)}  test: {len(test_labels)}")
split_sets = {
    "train": set(train_labels),
    "dev": set(dev_labels),
    "test": set(test_labels),
}
for (left_name, left_ids), (right_name, right_ids) in combinations(split_sets.items(), 2):
    overlap = left_ids & right_ids
    assert not overlap, f"overlapping IDs between {left_name} and {right_name}: {len(overlap)}"


def get_split(reply_id: str) -> str:
    if reply_id in dev_labels:
        return "dev"
    if reply_id in train_labels:
        return "train"
    if reply_id in test_labels:
        return "test"
    return "unknown"


# ── raw data directories ───────────────────────────────────────────────────────
REDDIT_DIRS: list[Path] = [
    ROOT / "data/raw/reddit-training-data",
    ROOT / "data/raw/reddit-dev-data",
    ROOT / "data/rumoureval2019/test-data/rumoureval-2019-test-data/reddit-test-data",
]
TWITTER_DIRS: list[Path] = [
    ROOT / "data/raw/twitter-english",
    ROOT / "data/rumoureval2019/test-data/rumoureval-2019-test-data/twitter-en-test-data",
]

# ── parsing ────────────────────────────────────────────────────────────────────

def parse_reddit(data_dir: Path) -> list[dict]:
    records = []
    if not data_dir.exists():
        print(f"  [skip] missing directory: {data_dir}")
        return records
    for thread_id in os.listdir(data_dir):
        thread_path = data_dir / thread_id
        if not thread_path.is_dir():
            continue

        src_dir = thread_path / "source-tweet"
        if not src_dir.exists():
            continue
        src_files = list(src_dir.iterdir())
        if not src_files:
            continue

        with open(src_files[0], encoding="utf-8") as f:
            src_data = json.load(f)
        src_body = src_data["data"]["children"][0]["data"]
        src_text = src_body.get("selftext") or src_body.get("title", "")
        src_id   = str(src_body.get("id", thread_id))

        records.append({
            "reply_id":  src_id,
            "thread_id": thread_id,
            "parent_id": None,
            "depth":     -1,
            "text":      src_text,
            "label":     label_map.get(thread_id),
            "platform":  "reddit",
            "split":     get_split(thread_id),
            "is_source": True,
        })

        rep_dir = thread_path / "replies"
        if not rep_dir.exists():
            continue
        for rep_file in rep_dir.iterdir():
            with open(rep_file, encoding="utf-8") as f:
                rep_data = json.load(f)
            rep_body = rep_data["data"]
            rep_id   = str(rep_body.get("id", rep_file.stem))
            records.append({
                "reply_id":  rep_id,
                "thread_id": thread_id,
                "parent_id": str(rep_body.get("parent_id", "").split("_")[-1]),
                "depth":     rep_body.get("depth", 0),
                "text":      rep_body.get("body", ""),
                "label":     label_map.get(rep_id),
                "platform":  "reddit",
                "split":     get_split(rep_id),
                "is_source": False,
            })
    return records


def parse_twitter(data_dir: Path) -> list[dict]:
    records = []
    if not data_dir.exists():
        print(f"  [skip] missing directory: {data_dir}")
        return records
    for event_id in os.listdir(data_dir):
        event_path = data_dir / event_id
        if not event_path.is_dir():
            continue
        for thread_id in os.listdir(event_path):
            thread_path = event_path / thread_id
            if not thread_path.is_dir():
                continue

            src_dir = thread_path / "source-tweet"
            if not src_dir.exists():
                continue
            src_files = list(src_dir.iterdir())
            if not src_files:
                continue

            with open(src_files[0], encoding="utf-8") as f:
                src_data = json.load(f)
            src_id   = str(src_data.get("id", thread_id))
            src_text = src_data.get("text", "")

            records.append({
                "reply_id":  src_id,
                "thread_id": thread_id,
                "parent_id": None,
                "depth":     -1,
                "text":      src_text,
                "label":     label_map.get(src_id),
                "platform":  "twitter",
                "split":     get_split(src_id),
                "is_source": True,
            })

            rep_dir = thread_path / "replies"
            if not rep_dir.exists():
                continue
            for rep_file in rep_dir.iterdir():
                with open(rep_file, encoding="utf-8") as f:
                    rep_data = json.load(f)
                rep_id    = str(rep_data.get("id", rep_file.stem))
                parent_id = str(rep_data.get("in_reply_to_status_id", ""))
                records.append({
                    "reply_id":  rep_id,
                    "thread_id": thread_id,
                    "parent_id": parent_id,
                    "depth":     0,
                    "text":      rep_data.get("text", ""),
                    "label":     label_map.get(rep_id),
                    "platform":  "twitter",
                    "split":     get_split(rep_id),
                    "is_source": False,
                })
    return records


# ── context condition builder ──────────────────────────────────────────────────

STANCE_KEYWORDS = [
    "confirmed", "confirm", "true", "false", "fake", "real",
    "denied", "deny", "hoax", "verified", "debunked", "proven",
    "official", "breaking", "fact",
]

LD_TEMPLATES: dict[str, str] = {
    "support": "This has been widely denied and officially debunked by experts.",
    "deny":    "Multiple credible sources have confirmed this to be completely true.",
    "query":   "There is absolutely no doubt about this. It has been verified.",
    "comment": "This claim has been officially confirmed as false by authorities.",
}


def _has_stance_keyword(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in STANCE_KEYWORDS)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_conditions(
    target: dict,
    thread_records: list[dict],
    id_to_record: dict[str, dict],
) -> dict | None:
    src = next((r for r in thread_records if r["is_source"]), None)
    if not src or not target["label"]:
        return None

    src_text  = src["text"]
    tgt_text  = target["text"]
    tgt_label = target["label"]
    parent    = id_to_record.get(target["parent_id"])

    # 1. reply_only
    reply_only = tgt_text

    # 2. useful
    if parent and not parent["is_source"]:
        useful = f"[Source] {src_text}\n[Context] {parent['text']}\n[Target] {tgt_text}"
    else:
        useful = f"[Source] {src_text}\n[Target] {tgt_text}"

    # 3. irrelevant
    ti_cands = [
        r for r in thread_records
        if r["label"] == "comment"
        and r["reply_id"] != target["reply_id"]
        and r["reply_id"] != target.get("parent_id")
        and not r["is_source"]
        and not _has_stance_keyword(r["text"])
    ]
    if ti_cands:
        ti_ctx = min(ti_cands, key=lambda r: _jaccard(r["text"], tgt_text))
        irrelevant = f"[Source] {src_text}\n[Context] {ti_ctx['text']}\n[Target] {tgt_text}"
        irrelevant_context_id = ti_ctx["reply_id"]
        valid_ti   = True
    else:
        irrelevant = None
        irrelevant_context_id = None
        valid_ti   = False

    # 4. conflicting
    conflict_pref: dict[str, list[str]] = {
        "support": ["deny"],
        "deny":    ["support"],
        "query":   ["support", "deny"],
        "comment": ["deny", "support"],
    }
    cc_cands = [
        r for r in thread_records
        if r["label"] in conflict_pref.get(tgt_label, [])
        and r["reply_id"] != target["reply_id"]
        and not r["is_source"]
        and len(r["text"].split()) >= 5
    ]
    if cc_cands:
        cc_ctx = max(
            cc_cands,
            key=lambda r: sum(1 for kw in STANCE_KEYWORDS if kw in r["text"].lower()),
        )
        conflicting = f"[Source] {src_text}\n[Context] {cc_ctx['text']}\n[Target] {tgt_text}"
        conflicting_context_id = cc_ctx["reply_id"]
        valid_cc    = True
    else:
        conflicting = None
        conflicting_context_id = None
        valid_cc    = False

    # 5. mixed
    if valid_cc:
        if parent and not parent["is_source"]:
            mixed = (
                f"[Source] {src_text}\n"
                f"[Context] {parent['text']}\n"
                f"[Misleading] {cc_ctx['text']}\n"
                f"[Target] {tgt_text}"
            )
        else:
            mixed = (
                f"[Source] {src_text}\n"
                f"[Misleading] {cc_ctx['text']}\n"
                f"[Target] {tgt_text}"
            )
    else:
        mixed = None

    # 6. lexical
    lexical = f"[Context] {LD_TEMPLATES[tgt_label]}\n[Target] {tgt_text}"

    return {
        "reply_id":    target["reply_id"],
        "thread_id":   target["thread_id"],
        "label":       tgt_label,
        "platform":    target["platform"],
        "split":       target["split"],
        "reply_only":  reply_only,
        "useful":      useful,
        "irrelevant":  irrelevant,
        "conflicting": conflicting,
        "mixed":       mixed,
        "lexical":     lexical,
        "useful_context_id": parent["reply_id"] if parent and not parent["is_source"] else None,
        "irrelevant_context_id": irrelevant_context_id,
        "conflicting_context_id": conflicting_context_id,
        "valid_ti":    valid_ti,
        "valid_cc":    valid_cc,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Parsing Reddit...")
    records: list[dict] = []
    for d in REDDIT_DIRS:
        r = parse_reddit(d)
        print(f"  {d.name}: {len(r)} records")
        records.extend(r)

    print("Parsing Twitter...")
    for d in TWITTER_DIRS:
        r = parse_twitter(d)
        print(f"  {d.name}: {len(r)} records")
        records.extend(r)

    print(f"\nTotal records: {len(records)}")
    print("Split dist (replies with labels):", Counter(
        r["split"] for r in records if not r["is_source"] and r["label"]
    ))

    threads: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        threads[r["thread_id"]].append(r)
    id_to_record: dict[str, dict] = {r["reply_id"]: r for r in records}

    print("\nBuilding context conditions...")
    dataset: list[dict] = []
    for target in records:
        if target["is_source"] or not target["label"]:
            continue
        result = build_conditions(target, threads[target["thread_id"]], id_to_record)
        if result:
            dataset.append(result)

    print(f"Dataset size: {len(dataset)}")
    for split in ("train", "dev", "test"):
        sub = [d for d in dataset if d["split"] == split]
        print(f"  {split}: {len(sub)}  label dist: {Counter(d['label'] for d in sub)}")

    out = ROOT / "data/processed/context_conditions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
