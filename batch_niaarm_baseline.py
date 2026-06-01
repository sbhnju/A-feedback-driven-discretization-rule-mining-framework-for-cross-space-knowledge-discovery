import re
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd

from niaarm import Dataset, get_rules


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def load_zone_data(file_path: Path, n_vars: int = 12) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine="openpyxl")

    if len(df.columns) < n_vars:
        num_cols = len(df.columns)
    else:
        num_cols = n_vars

    data = df.iloc[:, :num_cols].copy()
    data.columns = [f"Var{i+1}" for i in range(num_cols)]
    return data


def preprocess_data_like_proposed(data: pd.DataFrame):
    """
    与 proposed 原始代码尽量保持一致：
    - constant columns: nunique == 1
    - integer columns: dtype.kind in 'iu'
    """
    constant_cols = []
    integer_cols = []

    for col in data.columns:
        unique_vals = data[col].nunique()

        if unique_vals == 1:
            constant_cols.append(col)
        elif data[col].dtype.kind in "iu":
            integer_cols.append(col)

    return data, constant_cols, integer_cols


def infer_rule_length_from_text(rule_text: str) -> int:
    if rule_text is None or (isinstance(rule_text, float) and np.isnan(rule_text)):
        return 0

    s = str(rule_text).strip()
    if not s:
        return 0

    if "=>" in s:
        left, right = s.split("=>", 1)
    elif "->" in s:
        left, right = s.split("->", 1)
    elif "⇒" in s:
        left, right = s.split("⇒", 1)
    else:
        parts = re.split(r"\s*&\s*|\s*,\s*|\s+and\s+", s, flags=re.IGNORECASE)
        return len([p for p in parts if p.strip()])

    def count_side(x: str) -> int:
        parts = re.split(r"\s*&\s*|\s*,\s*|\s+and\s+", x.strip(), flags=re.IGNORECASE)
        return len([p for p in parts if p.strip()])

    return count_side(left) + count_side(right)


def infer_rule_length_from_columns(row: pd.Series) -> int:
    colmap = {normalize_column_name(c): c for c in row.index}

    ant_candidates = ["antecedent", "antecedents", "lhs"]
    con_candidates = ["consequent", "consequents", "rhs"]

    ant_col = next((colmap[c] for c in ant_candidates if c in colmap), None)
    con_col = next((colmap[c] for c in con_candidates if c in colmap), None)

    def count_items(x) -> int:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        s = str(x).strip()
        if not s:
            return 0
        parts = re.split(r"\s*&\s*|\s*,\s*|\s+and\s+", s, flags=re.IGNORECASE)
        return len([p for p in parts if p.strip()])

    if ant_col is not None or con_col is not None:
        ant_n = count_items(row[ant_col]) if ant_col is not None else 0
        con_n = count_items(row[con_col]) if con_col is not None else 0
        return ant_n + con_n

    rule_candidates = ["rule", "rules", "association_rule"]
    rule_col = next((colmap[c] for c in rule_candidates if c in colmap), None)
    if rule_col is not None:
        return infer_rule_length_from_text(row[rule_col])

    return 0


def find_metric_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    colmap = {normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in colmap:
            return colmap[cand]
    return None


def extract_variables_from_rule_text(text: str) -> Set[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set()
    return set(re.findall(r"Var\d+", str(text)))


def extract_variables_from_row(row: pd.Series) -> Set[str]:
    colmap = {normalize_column_name(c): c for c in row.index}

    ant_candidates = ["antecedent", "antecedents", "lhs"]
    con_candidates = ["consequent", "consequents", "rhs"]
    rule_candidates = ["rule", "rules", "association_rule"]

    variables = set()

    ant_col = next((colmap[c] for c in ant_candidates if c in colmap), None)
    con_col = next((colmap[c] for c in con_candidates if c in colmap), None)

    if ant_col is not None:
        variables.update(extract_variables_from_rule_text(row[ant_col]))
    if con_col is not None:
        variables.update(extract_variables_from_rule_text(row[con_col]))

    if variables:
        return variables

    rule_col = next((colmap[c] for c in rule_candidates if c in colmap), None)
    if rule_col is not None:
        variables.update(extract_variables_from_rule_text(row[rule_col]))

    return variables


def summarize_niaarm_rules(rules_csv_path: Path, zone_name: str) -> Dict:
    if not rules_csv_path.exists():
        return {
            "Zone": zone_name,
            "Valid_rules": 0,
            "Total_rules": 0,
            "Max_support": 0.0,
            "Mean_support": 0.0,
            "Mean_confidence": 0.0,
            "Mean_lift": 0.0,
            "Mean_rule_length": 0.0,
            "Variables_covered_n": 0,
            "Covered_variables": "",
            "Run_time_sec": np.nan
        }

    df = pd.read_csv(rules_csv_path)

    if df.empty:
        return {
            "Zone": zone_name,
            "Valid_rules": 0,
            "Total_rules": 0,
            "Max_support": 0.0,
            "Mean_support": 0.0,
            "Mean_confidence": 0.0,
            "Mean_lift": 0.0,
            "Mean_rule_length": 0.0,
            "Variables_covered_n": 0,
            "Covered_variables": "",
            "Run_time_sec": np.nan
        }

    support_col = find_metric_column(df, ["support"])
    confidence_col = find_metric_column(df, ["confidence"])
    lift_col = find_metric_column(df, ["lift"])

    rule_lengths = df.apply(infer_rule_length_from_columns, axis=1)

    covered_variables = set()
    for _, row in df.iterrows():
        covered_variables.update(extract_variables_from_row(row))

    mean_support = float(df[support_col].mean()) if support_col else np.nan
    max_support = float(df[support_col].max()) if support_col else np.nan
    mean_confidence = float(df[confidence_col].mean()) if confidence_col else np.nan
    mean_lift = float(df[lift_col].mean()) if lift_col else np.nan
    mean_rule_length = float(rule_lengths.mean()) if len(rule_lengths) > 0 else np.nan

    return {
        "Zone": zone_name,
        "Valid_rules": 1 if len(df) > 0 else 0,
        "Total_rules": int(len(df)),
        "Max_support": max_support if not np.isnan(max_support) else 0.0,
        "Mean_support": mean_support if not np.isnan(mean_support) else 0.0,
        "Mean_confidence": mean_confidence if not np.isnan(mean_confidence) else 0.0,
        "Mean_lift": mean_lift if not np.isnan(mean_lift) else 0.0,
        "Mean_rule_length": mean_rule_length if not np.isnan(mean_rule_length) else 0.0,
        "Variables_covered_n": len(covered_variables),
        "Covered_variables": ", ".join(sorted(covered_variables)),
        "Run_time_sec": np.nan
    }


def build_table15_niaarm_row(
    per_zone_summary: pd.DataFrame,
    all_covered_variables: Optional[Set[str]] = None,
    method_name: str = "NiaARM"
):
    valid_zone_count = int(per_zone_summary["Valid_rules"].sum())
    total_rules = int(per_zone_summary["Total_rules"].sum())
    max_support = float(per_zone_summary["Max_support"].max()) if len(per_zone_summary) > 0 else 0.0

    if total_rules > 0:
        mean_support = np.average(per_zone_summary["Mean_support"], weights=per_zone_summary["Total_rules"])
        mean_confidence = np.average(per_zone_summary["Mean_confidence"], weights=per_zone_summary["Total_rules"])
        mean_lift = np.average(per_zone_summary["Mean_lift"], weights=per_zone_summary["Total_rules"])
        mean_rule_length = np.average(per_zone_summary["Mean_rule_length"], weights=per_zone_summary["Total_rules"])
    else:
        mean_support = 0.0
        mean_confidence = 0.0
        mean_lift = 0.0
        mean_rule_length = 0.0

    variables_covered_n = len(all_covered_variables) if all_covered_variables is not None else 0

    return pd.DataFrame([{
        "Method": method_name,
        "Occupied zones with valid rules, n": valid_zone_count,
        "Total rules, n": total_rules,
        "Max support": max_support,
        "Mean support": mean_support,
        "Mean confidence": mean_confidence,
        "Mean lift": mean_lift,
        "Mean rule length": mean_rule_length,
        "Variables covered, n": variables_covered_n
    }])


def run_niaarm_for_zone(
    file_path: Path,
    zone_output_dir: Path,
    max_evals: int = 1000,
    seed: int = 1234,
    algorithm: str = "DifferentialEvolution",
    metrics: Tuple[str, ...] = ("support", "confidence")
) -> Dict:
    zone_name = file_path.stem
    safe_mkdir(zone_output_dir)

    # 读取数据
    data = load_zone_data(file_path, n_vars=12)

    # 与 proposed 保持一致的预处理
    data, constant_cols, integer_cols = preprocess_data_like_proposed(data)

    # constant columns 不参与规则挖掘
    data_for_rules = data.drop(columns=constant_cols, errors="ignore").copy()

    # 尽量保留整数属性
    for col in integer_cols:
        if col in data_for_rules.columns:
            try:
                data_for_rules[col] = pd.to_numeric(data_for_rules[col], errors="coerce").astype("Int64")
            except Exception:
                pass

    dataset = Dataset(data_for_rules)

    t0 = time.time()
    rules, run_time = get_rules(
        dataset,
        algorithm,
        metrics,
        max_evals=max_evals,
        seed=seed
    )
    elapsed = time.time() - t0

    rules_csv_path = zone_output_dir / f"{zone_name}_niaarm_rules.csv"
    rules.to_csv(str(rules_csv_path))

    data.to_excel(zone_output_dir / f"{zone_name}_input_data.xlsx", index=False)

    preprocess_info = pd.DataFrame({
        "Constant_columns": pd.Series(constant_cols),
        "Integer_columns": pd.Series(integer_cols)
    })
    preprocess_info.to_excel(zone_output_dir / f"{zone_name}_niaarm_preprocess_info.xlsx", index=False)

    summary = summarize_niaarm_rules(rules_csv_path, zone_name)
    summary["Run_time_sec"] = run_time if run_time is not None else elapsed

    pd.DataFrame([summary]).to_excel(
        zone_output_dir / f"{zone_name}_niaarm_summary.xlsx",
        index=False
    )

    return summary


def run_batch_niaarm(
    input_files,
    output_dir="niaarm_batch_output",
    max_evals: int = 1000,
    seed: int = 1234,
    algorithm: str = "DifferentialEvolution",
    metrics: Tuple[str, ...] = ("support", "confidence")
):
    output_dir = Path(output_dir)
    safe_mkdir(output_dir)

    per_zone_records = []
    all_covered_variables = set()

    for file_name in input_files:
        file_path = Path(file_name)
        zone_name = file_path.stem

        print("\n" + "=" * 72)
        print(f"Processing NiaARM zone file: {file_name}")
        print("=" * 72)

        if not file_path.exists():
            print(f"文件不存在，跳过: {file_name}")
            per_zone_records.append({
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Max_support": 0.0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Variables_covered_n": 0,
                "Covered_variables": "",
                "Run_time_sec": np.nan
            })
            continue

        try:
            zone_output_dir = output_dir / zone_name
            summary = run_niaarm_for_zone(
                file_path=file_path,
                zone_output_dir=zone_output_dir,
                max_evals=max_evals,
                seed=seed,
                algorithm=algorithm,
                metrics=metrics
            )

            per_zone_records.append(summary)

            if summary.get("Covered_variables", ""):
                vars_this_zone = {v.strip() for v in summary["Covered_variables"].split(",") if v.strip()}
                all_covered_variables.update(vars_this_zone)

            print(f"Zone = {summary['Zone']}")
            print(f"Valid rules = {summary['Valid_rules']}")
            print(f"Total rules = {summary['Total_rules']}")
            print(f"Max support = {summary['Max_support']:.6f}")
            print(f"Mean support = {summary['Mean_support']:.6f}")
            print(f"Mean confidence = {summary['Mean_confidence']:.6f}")
            print(f"Mean lift = {summary['Mean_lift']:.6f}")
            print(f"Mean rule length = {summary['Mean_rule_length']:.6f}")
            print(f"Variables covered = {summary['Variables_covered_n']}")
            print(f"Run time (s) = {summary['Run_time_sec']:.3f}")

        except Exception as e:
            print(f"处理失败: {file_name} | Error: {e}")
            per_zone_records.append({
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Max_support": 0.0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Variables_covered_n": 0,
                "Covered_variables": "",
                "Run_time_sec": np.nan
            })

    per_zone_summary = pd.DataFrame(per_zone_records)
    per_zone_summary.to_excel(output_dir / "niaarm_per_zone_summary.xlsx", index=False)

    covered_vars_df = pd.DataFrame({
        "Variable": sorted(list(all_covered_variables))
    })
    covered_vars_df.to_excel(output_dir / "niaarm_all_covered_variables.xlsx", index=False)

    table15_niaarm = build_table15_niaarm_row(
        per_zone_summary=per_zone_summary,
        all_covered_variables=all_covered_variables,
        method_name="NiaARM"
    )
    table15_niaarm.to_excel(output_dir / "Table15_niaarm_row.xlsx", index=False)

    with pd.ExcelWriter(output_dir / "niaarm_batch_summary.xlsx", engine="openpyxl") as writer:
        per_zone_summary.to_excel(writer, sheet_name="Per_Zone_Summary", index=False)
        covered_vars_df.to_excel(writer, sheet_name="All_Covered_Variables", index=False)
        table15_niaarm.to_excel(writer, sheet_name="Table15_NiaARM_Row", index=False)

    print("\n" + "=" * 72)
    print("NiaARM batch run finished.")
    print(f"Per-zone summary saved to: {output_dir / 'niaarm_per_zone_summary.xlsx'}")
    print(f"Covered variables saved to: {output_dir / 'niaarm_all_covered_variables.xlsx'}")
    print(f"Table 15 NiaARM row saved to: {output_dir / 'Table15_niaarm_row.xlsx'}")
    print(f"Combined summary saved to: {output_dir / 'niaarm_batch_summary.xlsx'}")
    print("=" * 72)

    return per_zone_summary, covered_vars_df, table15_niaarm


if __name__ == "__main__":
    input_files = [
        "Cluster 0.xlsx",
        "Cluster 1.xlsx",
        "Cluster 2.xlsx",
        "Cluster 3.xlsx",
        "Cluster 9.xlsx",
        "Cluster 12.xlsx",
        "Cluster 15.xlsx",
        "Cluster 21.xlsx",
        "Cluster 24.xlsx"
    ]

    per_zone_summary, covered_vars_df, table15_niaarm = run_batch_niaarm(
        input_files=input_files,
        output_dir="niaarm_batch_output",
        max_evals=1000,
        seed=1234,
        algorithm="DifferentialEvolution",
        metrics=("support", "confidence")
    )

    print("\nPer-zone summary:")
    print(per_zone_summary)

    print("\nAll covered variables:")
    print(covered_vars_df)

    print("\nTable 15 NiaARM row:")
    print(table15_niaarm)