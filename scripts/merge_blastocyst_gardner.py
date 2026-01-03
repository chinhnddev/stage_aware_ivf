"""
Merge Gardner silver/gold labels into blastocyst.csv and build leakage-safe splits.

Usage:
  python scripts/merge_blastocyst_gardner.py --output-dir data/metadata/merged_gardner
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.data.splits import split_by_group

MISSING_TOKENS = {"", "0", "ND", "NA", "N/A"}
ICM_TE_MAP = {1: "A", 2: "B", 3: "C"}


def _is_missing(value) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper() in MISSING_TOKENS


def _parse_int(value) -> Optional[int]:
    if _is_missing(value):
        return None
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    if num == 0:
        return None
    return num


def _parse_icm_te_letter(value) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C"}:
        return text
    num = _parse_int(value)
    if num is None:
        return None
    return ICM_TE_MAP.get(num)


def _build_grade(exp: Optional[int], icm: Optional[int], te: Optional[int]) -> Optional[str]:
    if exp is None or icm is None or te is None:
        return None
    icm_letter = ICM_TE_MAP.get(icm)
    te_letter = ICM_TE_MAP.get(te)
    if icm_letter is None or te_letter is None:
        return None
    return f"{exp}{icm_letter}{te_letter}"


def _extract_group_id(basename: str) -> Optional[str]:
    if not basename:
        return None
    text = str(basename).strip()
    if not text:
        return None
    prefix = text.split("_", 1)[0]
    return prefix or None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    drop_cols = [col for col in df.columns if not col or str(col).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _load_label_source(
    path: Path,
    exp_col: str,
    icm_col: str,
    te_col: str,
    source: str,
    conflicts: List[Dict],
) -> Dict[str, Dict]:
    df = pd.read_csv(path, sep=";")
    df = _normalize_columns(df)
    if "Image" not in df.columns:
        raise ValueError(f"{source} file missing Image column: {path}")
    for col in (exp_col, icm_col, te_col):
        if col not in df.columns:
            raise ValueError(f"{source} file missing {col} column: {path}")

    label_map: Dict[str, Dict] = {}
    for group_idx, (basename, group) in enumerate(df.groupby("Image", sort=False)):
        entries = []
        for row_idx, (_, row) in enumerate(group.iterrows()):
            exp = _parse_int(row.get(exp_col))
            icm = _parse_int(row.get(icm_col))
            te = _parse_int(row.get(te_col))
            entries.append(
                {
                    "exp": exp,
                    "icm": icm,
                    "te": te,
                    "grade": _build_grade(exp, icm, te),
                    "row_idx": row_idx,
                }
            )

        def _non_missing_set(field: str):
            values = [entry[field] for entry in entries if entry[field] is not None]
            return sorted({str(v) for v in values})

        for field in ("exp", "icm", "te"):
            values = _non_missing_set(field)
            if len(values) > 1:
                conflicts.append(
                    {
                        "basename": basename,
                        "source": source,
                        "field": field,
                        "values": "|".join(values),
                        "action": "dedupe_label_source",
                    }
                )

        best = max(
            entries,
            key=lambda entry: (sum(v is not None for v in (entry["exp"], entry["icm"], entry["te"])), -entry["row_idx"]),
        )
        label_map[basename] = best
    return label_map


def _dedupe_blastocyst(df: pd.DataFrame, silver_map: Dict[str, Dict], conflicts: List[Dict]) -> pd.DataFrame:
    dedup_rows = []
    dropped_duplicates = 0
    for basename, group in df.groupby("_basename", sort=False):
        group = group.sort_values("_row_index")
        if len(group) == 1:
            dedup_rows.append(group.iloc[0].to_dict())
            continue

        dropped_duplicates += len(group) - 1

        def _field_values(field: str):
            values = []
            for value in group[field]:
                if field in {"exp", "icm", "te"}:
                    norm = _parse_int(value)
                    if norm is not None:
                        values.append(str(norm))
                else:
                    if not _is_missing(value):
                        values.append(str(value).strip())
            return sorted(set(values))

        for field in ("exp", "icm", "te", "grade", "gardner"):
            if field not in group.columns:
                continue
            values = _field_values(field)
            if len(values) > 1:
                conflicts.append(
                    {
                        "basename": basename,
                        "source": "blastocyst_duplicate",
                        "field": field,
                        "values": "|".join(values),
                        "action": "dedupe_blastocyst",
                    }
                )

        if basename in silver_map:
            keep_row = group.iloc[0]
        else:
            def _score(row):
                score = 0
                for field in ("exp", "icm", "te", "grade", "gardner"):
                    if field in row and not _is_missing(row[field]):
                        score += 1
                return score

            best_idx = max(group.index, key=lambda idx: (_score(group.loc[idx]), -group.loc[idx]["_row_index"]))
            keep_row = group.loc[best_idx]
        dedup_rows.append(keep_row.to_dict())

    dedup_df = pd.DataFrame(dedup_rows)
    return dedup_df, dropped_duplicates


def _apply_updates(
    df: pd.DataFrame,
    label_map: Dict[str, Dict],
    source: str,
    mismatch_rows: List[Dict],
    updated_fields: Dict[str, int],
    skipped_missing: Dict[str, int],
) -> None:
    for idx, row in df.iterrows():
        basename = row["_basename"]
        if basename not in label_map:
            continue
        entry = label_map[basename]
        for field in ("exp", "icm", "te", "grade"):
            new_value = entry.get(field)
            if _is_missing(new_value):
                skipped_missing[field] += 1
                continue

            current_value = row.get(field)
            if field in {"exp", "icm", "te"}:
                current_norm = _parse_int(current_value)
                if current_norm != new_value:
                    if current_norm is not None:
                        mismatch_rows.append(
                            {
                                "basename": basename,
                                "field": field,
                                "existing": current_norm,
                                "incoming": new_value,
                                "source": source,
                            }
                        )
                    df.at[idx, field] = float(new_value)
                    df.at[idx, f"{field}_source"] = source
                    updated_fields[field] += 1
            else:
                current_norm = None if _is_missing(current_value) else str(current_value).strip()
                if current_norm != new_value:
                    if current_norm is not None:
                        mismatch_rows.append(
                            {
                                "basename": basename,
                                "field": field,
                                "existing": current_norm,
                                "incoming": new_value,
                                "source": source,
                            }
                        )
                    df.at[idx, field] = new_value
                    df.at[idx, f"{field}_source"] = source
                    updated_fields[field] += 1


def _backfill_grade_from_components(df: pd.DataFrame, updated_fields: Dict[str, int]) -> None:
    for idx, row in df.iterrows():
        if not _is_missing(row.get("grade")):
            continue
        exp = _parse_int(row.get("exp"))
        if exp is None:
            continue
        if exp < 3:
            continue
        icm_letter = _parse_icm_te_letter(row.get("icm"))
        te_letter = _parse_icm_te_letter(row.get("te"))
        if icm_letter is None or te_letter is None:
            continue
        grade = f"{exp}{icm_letter}{te_letter}"
        df.at[idx, "grade"] = grade
        df.at[idx, "grade_source"] = "derived_from_components"
        updated_fields["grade"] += 1


def _normalize_biological_constraints(df: pd.DataFrame) -> Dict[str, int]:
    exp_valid = df["exp"].apply(_parse_int)
    icm_valid = df["icm"].apply(_parse_int)
    te_valid = df["te"].apply(_parse_int)
    before_invalid = ((exp_valid.isna()) | (exp_valid < 3)) & (
        icm_valid.isin([1, 2, 3]) | te_valid.isin([1, 2, 3])
    )
    before_invalid_count = int(before_invalid.sum())

    for idx, row in df.iterrows():
        exp = _parse_int(row.get("exp"))
        if exp is None:
            df.at[idx, "exp"] = 0.0
            df.at[idx, "exp_source"] = "missing"
        if exp is None or exp < 3:
            df.at[idx, "icm"] = 0.0
            df.at[idx, "te"] = 0.0
            df.at[idx, "icm_source"] = "not_defined_by_stage"
            df.at[idx, "te_source"] = "not_defined_by_stage"
            continue

        icm = _parse_int(row.get("icm"))
        te = _parse_int(row.get("te"))
        if icm not in {1, 2, 3}:
            df.at[idx, "icm"] = 0.0
            df.at[idx, "icm_source"] = "missing"
        if te not in {1, 2, 3}:
            df.at[idx, "te"] = 0.0
            df.at[idx, "te_source"] = "missing"

    exp_valid = df["exp"].apply(_parse_int)
    icm_valid = df["icm"].apply(_parse_int)
    te_valid = df["te"].apply(_parse_int)
    after_invalid = ((exp_valid.isna()) | (exp_valid < 3)) & (
        icm_valid.isin([1, 2, 3]) | te_valid.isin([1, 2, 3])
    )
    after_invalid_count = int(after_invalid.sum())

    df["exp_defined"] = exp_valid.isin([1, 2, 3, 4, 5, 6]).astype(int)
    df["icm_defined"] = ((exp_valid >= 3) & icm_valid.isin([1, 2, 3])).astype(int)
    df["te_defined"] = ((exp_valid >= 3) & te_valid.isin([1, 2, 3])).astype(int)

    return {
        "before_invalid_icm_te": before_invalid_count,
        "after_invalid_icm_te": after_invalid_count,
        "exp_defined": int(df["exp_defined"].sum()),
        "icm_defined": int(df["icm_defined"].sum()),
        "te_defined": int(df["te_defined"].sum()),
    }


def _count_label_coverage(df: pd.DataFrame) -> Dict[str, int]:
    exp_valid = df["exp"].apply(_parse_int)
    exp_labeled = exp_valid.notna().sum()
    exp_ge3 = (exp_valid >= 3).sum()
    if "icm_defined" in df.columns and "te_defined" in df.columns:
        icm_mask = df["icm_defined"].sum()
        te_mask = df["te_defined"].sum()
    else:
        icm_valid = df["icm"].apply(_parse_icm_te_letter)
        te_valid = df["te"].apply(_parse_icm_te_letter)
        icm_mask = ((exp_valid >= 3) & icm_valid.notna()).sum()
        te_mask = ((exp_valid >= 3) & te_valid.notna()).sum()
    return {
        "rows_with_exp": int(exp_labeled),
        "rows_with_exp_ge3": int(exp_ge3),
        "rows_with_icm_mask": int(icm_mask),
        "rows_with_te_mask": int(te_mask),
    }


def _class_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    exp_valid = df["exp"].apply(_parse_int)
    icm_valid = df["icm"].apply(_parse_icm_te_letter)
    te_valid = df["te"].apply(_parse_icm_te_letter)
    if "icm_defined" in df.columns:
        icm_mask = df["icm_defined"] == 1
    else:
        icm_mask = exp_valid >= 3
    if "te_defined" in df.columns:
        te_mask = df["te_defined"] == 1
    else:
        te_mask = exp_valid >= 3
    icm_counts = (
        icm_valid[icm_mask & icm_valid.notna()].value_counts().reindex(["A", "B", "C"], fill_value=0).to_dict()
    )
    te_counts = (
        te_valid[te_mask & te_valid.notna()].value_counts().reindex(["A", "B", "C"], fill_value=0).to_dict()
    )
    return {"icm": icm_counts, "te": te_counts}


def _assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_col: Optional[str]) -> None:
    train_bases = set(train_df["_basename"])
    val_bases = set(val_df["_basename"])
    test_bases = set(test_df["_basename"])
    if train_bases & val_bases or train_bases & test_bases or val_bases & test_bases:
        raise ValueError("Basename overlap detected between train/val/test_gold.")

    if not group_col:
        raise ValueError("group_col is required for overlap check.")
    for name, df in (("train", train_df), ("val", val_df), ("test_gold", test_df)):
        if group_col not in df.columns:
            raise ValueError(f"group_col={group_col} missing in {name} split.")

    def _groups(df: pd.DataFrame) -> set:
        series = df[group_col].astype(str)
        if series.isna().any() or (series.str.strip() == "").any():
            raise ValueError(f"group_col={group_col} has missing values; ensure group_id coverage is complete.")
        return set(series)

    train_groups = _groups(train_df)
    val_groups = _groups(val_df)
    test_groups = _groups(test_df)

    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise ValueError(f"Group overlap detected across splits: {sorted(list(overlap))[:3]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Gardner labels into blastocyst.csv.")
    parser.add_argument("--blastocyst-csv", default="data/metadata/blastocyst.csv")
    parser.add_argument("--silver-csv", default="data/blastocyst_Dataset/Gardner_train_silver.csv")
    parser.add_argument("--gold-csv", default="data/blastocyst_Dataset/Gardner_test_gold_onlyGardnerScores.csv")
    parser.add_argument("--blastocyst-config", default="configs/data/blastocyst.yaml")
    parser.add_argument("--output-dir", default="data/metadata/merged_gardner")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.blastocyst_config)
    split_cfg = cfg.get("split", {}) if cfg is not None else {}
    group_col = split_cfg.get("group_col")
    val_ratio = float(split_cfg.get("val_ratio", 0.2))

    blast_df = pd.read_csv(args.blastocyst_csv)
    blast_df["_row_index"] = list(range(len(blast_df)))
    blast_df["_basename"] = (
        blast_df["image_path"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
    )
    blast_df["group_id"] = blast_df["_basename"].apply(_extract_group_id)
    blast_df["group_id"] = blast_df["group_id"].fillna(
        blast_df["_row_index"].apply(lambda idx: f"unknown_{idx}")
    )
    blast_df["group_id"] = blast_df["group_id"].astype(str)
    missing_group_id = (blast_df["group_id"].isna() | (blast_df["group_id"].str.strip() == "")).sum()
    if missing_group_id:
        raise ValueError(f"group_id missing after fallback: {missing_group_id}")
    if "embryo_id" in blast_df.columns:
        def _fill_embryo_id(row):
            current = row.get("embryo_id")
            if not _is_missing(current):
                return current
            group_id = row.get("group_id")
            try:
                return int(group_id)
            except (TypeError, ValueError):
                return current
        blast_df["embryo_id"] = blast_df.apply(_fill_embryo_id, axis=1)

    conflicts: List[Dict] = []
    mismatch_rows: List[Dict] = []
    updated_fields = {field: 0 for field in ("exp", "icm", "te", "grade")}
    skipped_missing = {field: 0 for field in ("exp", "icm", "te", "grade")}

    silver_map = _load_label_source(
        Path(args.silver_csv),
        exp_col="EXP_silver",
        icm_col="ICM_silver",
        te_col="TE_silver",
        source="silver",
        conflicts=conflicts,
    )
    gold_map = _load_label_source(
        Path(args.gold_csv),
        exp_col="EXP_gold",
        icm_col="ICM_gold",
        te_col="TE_gold",
        source="gold",
        conflicts=conflicts,
    )

    dedup_df, dropped_duplicates = _dedupe_blastocyst(blast_df, silver_map, conflicts)
    for field in ("exp", "icm", "te", "grade"):
        source_col = f"{field}_source"
        dedup_df[source_col] = dedup_df[field].apply(lambda v: "original" if not _is_missing(v) else "missing")
    _apply_updates(dedup_df, silver_map, "silver", mismatch_rows, updated_fields, skipped_missing)
    _apply_updates(dedup_df, gold_map, "gold", mismatch_rows, updated_fields, skipped_missing)
    _backfill_grade_from_components(dedup_df, updated_fields)
    dedup_df["grade_source"] = dedup_df["grade_source"].fillna("missing")
    before_snapshot = dedup_df[
        ["_basename", "exp", "icm", "te", "exp_source", "icm_source", "te_source", "grade", "grade_source"]
    ].copy()
    pre_exp = before_snapshot["exp"].apply(_parse_int)
    pre_icm = before_snapshot["icm"].apply(_parse_int)
    pre_te = before_snapshot["te"].apply(_parse_int)
    pre_invalid = ((pre_exp.isna()) | (pre_exp < 3)) & (pre_icm.isin([1, 2, 3]) | pre_te.isin([1, 2, 3]))
    example_indices = list(pre_invalid[pre_invalid].head(3).index)

    normalization_stats = _normalize_biological_constraints(dedup_df)
    bad_after = ((dedup_df["exp_defined"] == 0) | (dedup_df["exp"].apply(_parse_int) < 3)) & (
        (dedup_df["icm_defined"] == 1) | (dedup_df["te_defined"] == 1)
    )
    if bad_after.any():
        raise ValueError("Normalization failed: exp<3 rows still have icm_defined/te_defined.")

    if example_indices:
        print("Examples before/after normalization:")
        for idx in example_indices:
            before = before_snapshot.loc[idx]
            after = dedup_df.loc[idx]
            print(
                f"{before['_basename']} before(exp={before['exp']} icm={before['icm']} te={before['te']} "
                f"exp_source={before['exp_source']} icm_source={before['icm_source']} te_source={before['te_source']}) "
                f"-> after(exp={after['exp']} icm={after['icm']} te={after['te']} "
                f"exp_source={after['exp_source']} icm_source={after['icm_source']} te_source={after['te_source']})"
            )

    silver_matched = dedup_df["_basename"].isin(silver_map).sum()
    gold_matched = dedup_df["_basename"].isin(gold_map).sum()

    gold_base_df = dedup_df[dedup_df["_basename"].isin(gold_map)].copy()
    gold_group_ids = set(gold_base_df["group_id"].astype(str))
    gold_df = dedup_df[dedup_df["group_id"].astype(str).isin(gold_group_ids)].copy()
    non_gold_df = dedup_df[~dedup_df["group_id"].astype(str).isin(gold_group_ids)].copy()
    gold_group_rows = len(gold_df)

    group_col_for_split = "group_id"
    splits = split_by_group(
        non_gold_df,
        group_col=group_col_for_split,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=args.seed,
    )
    train_df = splits["train"].copy()
    val_df = splits["val"].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    gold_df["split"] = "test_gold"

    _assert_no_overlap(train_df, val_df, gold_df, group_col_for_split)

    split_map = {}
    split_map.update({row["_basename"]: "train" for _, row in train_df.iterrows()})
    split_map.update({row["_basename"]: "val" for _, row in val_df.iterrows()})
    split_map.update({row["_basename"]: "test_gold" for _, row in gold_df.iterrows()})
    dedup_df["split"] = dedup_df["_basename"].map(split_map)

    def _drop_helpers(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[col for col in ("_basename", "_row_index") if col in df.columns])

    blastocyst_merged = _drop_helpers(dedup_df)
    blastocyst_merged.to_csv(output_dir / "blastocyst_merged.csv", index=False)
    blastocyst_merged.to_csv(output_dir / "blastocyst_merged_v2.csv", index=False)
    _drop_helpers(train_df).to_csv(output_dir / "train.csv", index=False)
    _drop_helpers(val_df).to_csv(output_dir / "val.csv", index=False)
    _drop_helpers(gold_df).to_csv(output_dir / "test_gold.csv", index=False)

    merge_conflicts = pd.DataFrame(conflicts)
    if not merge_conflicts.empty:
        merge_conflicts.to_csv(output_dir / "merge_conflicts.csv", index=False)
    else:
        pd.DataFrame(columns=["basename", "source", "field", "values", "action"]).to_csv(
            output_dir / "merge_conflicts.csv", index=False
        )

    mismatch_df = pd.DataFrame(mismatch_rows)
    if not mismatch_df.empty:
        mismatch_df.to_csv(output_dir / "mismatch_rows.csv", index=False)
    else:
        pd.DataFrame(columns=["basename", "field", "existing", "incoming", "source"]).to_csv(
            output_dir / "mismatch_rows.csv", index=False
        )

    report = {
        "blastocyst_rows": int(len(blast_df)),
        "dedup_rows": int(len(dedup_df)),
        "dropped_duplicates": int(dropped_duplicates),
        "silver_matched": int(silver_matched),
        "gold_matched": int(gold_matched),
        "updated_fields": updated_fields,
        "skipped_missing": skipped_missing,
        "conflicts": int(len(conflicts)),
        "mismatch_rows": int(len(mismatch_rows)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_gold_rows": int(len(gold_df)),
    }
    (output_dir / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    coverage = _count_label_coverage(dedup_df)
    class_dist = {
        "train": _class_distribution(train_df),
        "val": _class_distribution(val_df),
        "test_gold": _class_distribution(gold_df),
    }
    report_v2 = {
        "total_rows": int(len(dedup_df)),
        "duplicates_removed": int(dropped_duplicates),
        "conflicts_resolved": int(len(conflicts)),
        "group_id_missing": int((dedup_df["group_id"].isna() | (dedup_df["group_id"].str.strip() == "")).sum()),
        "group_id_unique": int(dedup_df["group_id"].nunique()),
        "gold_group_ids": int(len(gold_group_ids)),
        "gold_base_rows": int(gold_matched),
        "gold_group_rows": int(gold_group_rows),
        "removed_due_to_gold_group": int(len(dedup_df) - len(non_gold_df) - len(gold_df)),
        "before_invalid_icm_te": normalization_stats["before_invalid_icm_te"],
        "after_invalid_icm_te": normalization_stats["after_invalid_icm_te"],
        "exp_defined": normalization_stats["exp_defined"],
        "icm_defined": normalization_stats["icm_defined"],
        "te_defined": normalization_stats["te_defined"],
        **coverage,
        "class_distribution": class_dist,
    }
    (output_dir / "merge_report_v2.json").write_text(json.dumps(report_v2, indent=2), encoding="utf-8")

    print("Merge complete.")
    print(f"Matched: silver={silver_matched} gold={gold_matched} gold_group_rows={gold_group_rows}")
    print(f"Updated fields: {updated_fields}")
    print(f"Skipped missing: {skipped_missing}")
    print(f"Conflicts: {len(conflicts)} Dropped duplicates: {dropped_duplicates}")


if __name__ == "__main__":
    main()
