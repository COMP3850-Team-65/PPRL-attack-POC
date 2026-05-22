#!/usr/bin/env python3
"""Check patient disjointness between target and shadow data splits."""

import os
import pandas as pd

DATA_DIR = "data/external/BaselineDataSplits"

FILES = {
    "target_train": os.path.join(DATA_DIR, "target_train.csv"),
    "target_test": os.path.join(DATA_DIR, "target_test.csv"),
    "shadow_train": os.path.join(DATA_DIR, "shadow_train.csv"),
    "shadow_test": os.path.join(DATA_DIR, "shadow_test.csv"),
}


def extract_patient_ids(df: pd.DataFrame) -> set:
    ids = set(df["uid1"].unique())
    ids.update(df["uid2"].unique())
    return ids


def main():
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run from the project root.")
            return

    split_ids = {}
    for name, path in FILES.items():
        df = pd.read_csv(path)
        ids = extract_patient_ids(df)
        split_ids[name] = ids
        print(f"{name}: {len(df)} pairs, {len(ids)} unique patients")

    print()

    target_all = split_ids["target_train"] | split_ids["target_test"]
    shadow_all = split_ids["shadow_train"] | split_ids["shadow_test"]
    all_patients = target_all | shadow_all

    overlap = target_all & shadow_all
    print(f"Target total unique patients:   {len(target_all)}")
    print(f"Shadow total unique patients:   {len(shadow_all)}")
    print(f"Union:                          {len(all_patients)}")
    print(f"Overlap (target ∩ shadow):      {len(overlap)}")
    if len(target_all) > 0:
        print(f"  {len(overlap)} / {len(target_all)} target patients also in shadow ({100 * len(overlap) / len(target_all):.1f}%)")
    if len(shadow_all) > 0:
        print(f"  {len(overlap)} / {len(shadow_all)} shadow patients also in target ({100 * len(overlap) / len(shadow_all):.1f}%)")

    if overlap:
        print(f"\nOverlapping patient IDs (first 20): {sorted(overlap)[:20]}")
        if len(overlap) > 20:
            print(f"  ({len(overlap)} overlapping — showing first 20)")
    else:
        print("\nNo patient overlap.")

    print()

    comparisons = [
        ("target_train", "shadow_train"),
        ("target_train", "shadow_test"),
        ("target_test", "shadow_train"),
        ("target_test", "shadow_test"),
    ]
    print("Cross-table:")
    for name_a, name_b in comparisons:
        overlap_ab = split_ids[name_a] & split_ids[name_b]
        total_ab = split_ids[name_a] | split_ids[name_b]
        pct_a = 100 * len(overlap_ab) / len(split_ids[name_a]) if split_ids[name_a] else 0
        pct_b = 100 * len(overlap_ab) / len(split_ids[name_b]) if split_ids[name_b] else 0
        print(f"  {name_a} ∩ {name_b}: {len(overlap_ab)} patients ({pct_a:.1f}% of {name_a}, {pct_b:.1f}% of {name_b})")


if __name__ == "__main__":
    main()
