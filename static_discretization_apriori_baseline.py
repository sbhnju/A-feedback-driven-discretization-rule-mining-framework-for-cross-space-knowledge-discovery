import re
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore", category=RuntimeWarning)


class StaticDiscretizationAprioriBaseline:
    """
    Conventional static discretization + Apriori baseline

    统一口径：
    - constant columns: nunique == 1
    - integer columns: dtype.kind in 'iu'
    - constant columns 不参与规则挖掘
    - integer columns 单独处理
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fixed_bins: int = 5,
        integer_unique_threshold: int = 10,
        min_support_thresh: float = 0.1,
        rule_conf_thresh: float = 0.5,
        random_state: int = 42,
        max_len: int = 5
    ):
        self.raw_data = data.copy()
        self.fixed_bins = fixed_bins
        self.integer_unique_threshold = integer_unique_threshold
        self.min_support_thresh = min_support_thresh
        self.rule_conf_thresh = rule_conf_thresh
        self.random_state = random_state
        self.max_len = max_len

        self.original_data, self.constant_columns, self.integer_columns = self.preprocess_data(
            self.raw_data.copy()
        )

        self.discretized_data = None
        self.bin_info = None
        self.transactions = None
        self.rules = None

    def preprocess_data(self, data: pd.DataFrame):
        """
        与 proposed 原始代码保持一致：
        - unique_vals == 1 -> constant
        - dtype.kind in 'iu' -> integer
        """
        constant_cols = []
        int_cols = []

        for col in data.columns:
            unique_vals = data[col].nunique()

            if unique_vals == 1:
                constant_cols.append(col)
            elif data[col].dtype.kind in "iu":
                int_cols.append(col)

        return data, constant_cols, int_cols

    def fixed_discretize(self):
        discretized_data = self.original_data.copy()
        bin_info = {}

        # 1) constant columns
        for col in self.constant_columns:
            constant_value = self.original_data[col].iloc[0]
            bin_label = f"{col}_Constant"
            discretized_data[col] = bin_label
            bin_info[col] = {
                "strategy": "constant",
                "n_bins": 1,
                "boundaries": [constant_value, constant_value],
                "value_ranges": {bin_label: (constant_value, constant_value)}
            }

        # 2) non-constant columns
        non_constant_cols = [c for c in self.original_data.columns if c not in self.constant_columns]

        for col in non_constant_cols:
            series = pd.to_numeric(self.original_data[col], errors="coerce")
            unique_vals = series.dropna().nunique()

            # integer columns: special handling
            if col in self.integer_columns and unique_vals <= self.integer_unique_threshold:
                unique_sorted = sorted(series.dropna().unique())
                val_to_label = {val: f"{col}_Int{i}" for i, val in enumerate(unique_sorted)}

                discretized_col = series.map(val_to_label).astype(object)
                discretized_col[pd.isna(discretized_col)] = f"{col}_Missing"

                discretized_data[col] = discretized_col
                bin_info[col] = {
                    "strategy": "integer_unique",
                    "n_bins": len(unique_sorted),
                    "boundaries": unique_sorted,
                    "value_ranges": {val_to_label[val]: (val, val) for val in unique_sorted}
                }
                continue

            # other variables: fixed-bin KMeans
            discretized_data, bin_info = self.kmeans_discretize_col(
                discretized_data=discretized_data,
                bin_info=bin_info,
                col=col,
                fixed_bins=self.fixed_bins,
                series=series
            )

        self.discretized_data = discretized_data
        self.bin_info = bin_info
        return discretized_data, bin_info

    def kmeans_discretize_col(self, discretized_data, bin_info, col, fixed_bins, series):
        valid = series.dropna()

        if len(valid) == 0:
            bin_label = f"{col}_MissingOnly"
            discretized_data[col] = bin_label
            bin_info[col] = {
                "strategy": "missing_only",
                "n_bins": 1,
                "boundaries": [np.nan, np.nan],
                "value_ranges": {bin_label: (np.nan, np.nan)}
            }
            return discretized_data, bin_info

        unique_vals = valid.nunique()
        k = min(fixed_bins, unique_vals)

        if k <= 1:
            min_val = valid.min()
            max_val = valid.max()
            bin_label = f"{col}_SingleValue"
            discretized_col = pd.Series([bin_label] * len(series), index=series.index, dtype=object)
            discretized_col[pd.isna(series)] = f"{col}_Missing"

            discretized_data[col] = discretized_col
            bin_info[col] = {
                "strategy": "single_bin",
                "n_bins": 1,
                "boundaries": [min_val, max_val],
                "value_ranges": {bin_label: (min_val, max_val)}
            }
            return discretized_data, bin_info

        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(valid.values.reshape(-1, 1))

            kmeans = KMeans(n_clusters=k, n_init=20, random_state=self.random_state)
            labels_valid = kmeans.fit_predict(scaled)

            clusters = {}
            for lbl, val in zip(labels_valid, valid.values):
                clusters.setdefault(lbl, []).append(val)

            cluster_ranges = {
                lbl: (float(np.min(vals)), float(np.max(vals)))
                for lbl, vals in clusters.items()
            }

            sorted_clusters = sorted(cluster_ranges.items(), key=lambda x: x[1][0])

            label_map = {}
            value_ranges = {}
            boundaries = [sorted_clusters[0][1][0]]

            for new_idx, (old_lbl, (min_val, max_val)) in enumerate(sorted_clusters):
                new_label = f"{col}_Bin{new_idx}"
                label_map[old_lbl] = new_label
                value_ranges[new_label] = (min_val, max_val)
                boundaries.append(max_val)

            discretized_col = pd.Series(index=series.index, dtype=object)
            for idx, old_lbl in zip(valid.index, labels_valid):
                discretized_col.loc[idx] = label_map[old_lbl]
            discretized_col[pd.isna(series)] = f"{col}_Missing"

            discretized_data[col] = discretized_col
            bin_info[col] = {
                "strategy": "fixed_kmeans",
                "n_bins": k,
                "boundaries": boundaries,
                "value_ranges": value_ranges
            }
            return discretized_data, bin_info

        except Exception:
            return self.qcut_fallback(discretized_data, bin_info, col, fixed_bins, series)

    def qcut_fallback(self, discretized_data, bin_info, col, fixed_bins, series):
        valid = series.dropna()
        unique_vals = valid.nunique()
        k = min(fixed_bins, unique_vals)

        if k <= 1:
            min_val = valid.min()
            max_val = valid.max()
            bin_label = f"{col}_SingleValue"
            discretized_col = pd.Series([bin_label] * len(series), index=series.index, dtype=object)
            discretized_col[pd.isna(series)] = f"{col}_Missing"

            discretized_data[col] = discretized_col
            bin_info[col] = {
                "strategy": "single_bin_fallback",
                "n_bins": 1,
                "boundaries": [min_val, max_val],
                "value_ranges": {bin_label: (min_val, max_val)}
            }
            return discretized_data, bin_info

        labels = [f"{col}_Bin{i}" for i in range(k)]
        discretized_valid, bins = pd.qcut(
            valid,
            q=k,
            labels=labels,
            retbins=True,
            duplicates="drop"
        )

        discretized_col = pd.Series(index=series.index, dtype=object)
        discretized_col.loc[valid.index] = discretized_valid.astype(str)
        discretized_col[pd.isna(series)] = f"{col}_Missing"

        value_ranges = {}
        for label in discretized_valid.astype(str).unique():
            vals = valid[discretized_valid.astype(str) == label]
            value_ranges[label] = (float(vals.min()), float(vals.max()))

        discretized_data[col] = discretized_col
        bin_info[col] = {
            "strategy": "qcut_fallback",
            "n_bins": len(value_ranges),
            "boundaries": bins.tolist(),
            "value_ranges": value_ranges
        }
        return discretized_data, bin_info

    def prepare_transactions(self, discretized_data: pd.DataFrame):
        """
        与 proposed 原始代码一致：
        - 跳过 constant columns
        """
        transactions = []
        for _, row in discretized_data.iterrows():
            transaction = []
            for col, val in row.items():
                if col in self.constant_columns:
                    continue
                if pd.notna(val):
                    transaction.append(str(val))
            transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def mine_association_rules(self, transactions: pd.DataFrame):
        if len([c for c in self.original_data.columns if c not in self.constant_columns]) == 0:
            return pd.DataFrame()

        if transactions.empty or transactions.shape[1] == 0:
            return pd.DataFrame()

        min_support = max(self.min_support_thresh, 50 / len(transactions))

        try:
            frequent_itemsets = apriori(
                transactions,
                min_support=min_support,
                use_colnames=True,
                max_len=self.max_len
            )

            if frequent_itemsets.empty:
                return pd.DataFrame()

            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.rule_conf_thresh
            )

            if not rules.empty:
                rules["lift"] = rules["confidence"] / rules["support"]
                rules["antecedent_len"] = rules["antecedents"].apply(len)
                rules["consequent_len"] = rules["consequents"].apply(len)
                rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

            return rules

        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _extract_variable_name(item: str):
        if item is None:
            return None
        m = re.match(r"^(Var\d+)_", str(item).strip())
        return m.group(1) if m else None

    def extract_variables_from_rules(self, rules: pd.DataFrame):
        variables = set()

        if rules is None or rules.empty:
            return variables

        for _, row in rules.iterrows():
            ants = row.get("antecedents", [])
            cons = row.get("consequents", [])

            for item in ants:
                var_name = self._extract_variable_name(item)
                if var_name:
                    variables.add(var_name)

            for item in cons:
                var_name = self._extract_variable_name(item)
                if var_name:
                    variables.add(var_name)

        return variables

    def summarize_rules(self, rules: pd.DataFrame, zone_name: str):
        covered_vars = self.extract_variables_from_rules(rules)

        if rules is None or rules.empty:
            return {
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Max_support": 0.0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Variables_covered_n": 0
            }

        return {
            "Zone": zone_name,
            "Valid_rules": 1,
            "Total_rules": int(len(rules)),
            "Max_support": float(rules["support"].max()),
            "Mean_support": float(rules["support"].mean()),
            "Mean_confidence": float(rules["confidence"].mean()),
            "Mean_lift": float(rules["lift"].mean()),
            "Mean_rule_length": float(rules["rule_len"].mean()),
            "Variables_covered_n": len(covered_vars)
        }

    def save_zone_results(
        self,
        original_data: pd.DataFrame,
        discretized_data: pd.DataFrame,
        bin_info: dict,
        rules: pd.DataFrame,
        zone_output_dir: Path,
        zone_name: str
    ):
        zone_output_dir.mkdir(parents=True, exist_ok=True)

        combined_data = pd.concat(
            [original_data, discretized_data.add_prefix("Disc_")],
            axis=1
        )
        combined_data.to_excel(zone_output_dir / f"{zone_name}_static_discretized_data.xlsx", index=False)

        config_rows = []
        for col, info in bin_info.items():
            config_rows.append({
                "Variable": col,
                "Strategy": info.get("strategy", ""),
                "Bins": info.get("n_bins", ""),
                "Boundaries": str(info.get("boundaries", "")),
                "Value_Ranges": str(info.get("value_ranges", ""))
            })
        pd.DataFrame(config_rows).to_excel(
            zone_output_dir / f"{zone_name}_static_bins_config.xlsx", index=False
        )

        # 保存预处理信息，便于复核
        preprocess_info = pd.DataFrame({
            "Constant_columns": pd.Series(self.constant_columns),
            "Integer_columns": pd.Series(self.integer_columns)
        })
        preprocess_info.to_excel(
            zone_output_dir / f"{zone_name}_static_preprocess_info.xlsx",
            index=False
        )

        if rules is not None and not rules.empty:
            rules_to_save = rules.copy()
            rules_to_save["antecedents"] = rules_to_save["antecedents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            rules_to_save["consequents"] = rules_to_save["consequents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            rules_to_save.to_excel(
                zone_output_dir / f"{zone_name}_static_association_rules.xlsx", index=False
            )

            high_support_rules = rules_to_save[rules_to_save["support"] > 0.5]
            high_support_rules.to_excel(
                zone_output_dir / f"{zone_name}_static_high_support_rules.xlsx", index=False
            )

    def run(self):
        discretized_data, bin_info = self.fixed_discretize()
        transactions = self.prepare_transactions(discretized_data)
        rules = self.mine_association_rules(transactions)

        self.discretized_data = discretized_data
        self.bin_info = bin_info
        self.transactions = transactions
        self.rules = rules

        return discretized_data, bin_info, rules


def load_zone_data(file_path: Path, n_vars: int = 12) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    if len(df.columns) < n_vars:
        num_cols = len(df.columns)
    else:
        num_cols = n_vars

    data = df.iloc[:, :num_cols].copy()
    data.columns = [f"Var{i+1}" for i in range(num_cols)]
    return data


def build_table15_static_row(
    per_zone_summary: pd.DataFrame,
    all_covered_variables=None,
    method_name: str = "Static discretization + Apriori"
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


def run_batch_static_baseline(
    input_files,
    output_dir="static_baseline_batch_output",
    fixed_bins=5,
    integer_unique_threshold=10,
    min_support_thresh=0.1,
    rule_conf_thresh=0.5,
    random_state=42,
    max_len=5
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_zone_records = []
    all_covered_variables = set()

    for file_name in input_files:
        file_path = Path(file_name)
        zone_name = file_path.stem

        print("\n" + "=" * 70)
        print(f"Processing zone file: {file_name}")
        print("=" * 70)

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
                "Variables_covered_n": 0
            })
            continue

        try:
            data = load_zone_data(file_path, n_vars=12)
            print(f"加载成功: {file_name} | shape = {data.shape}")

            baseline = StaticDiscretizationAprioriBaseline(
                data=data,
                fixed_bins=fixed_bins,
                integer_unique_threshold=integer_unique_threshold,
                min_support_thresh=min_support_thresh,
                rule_conf_thresh=rule_conf_thresh,
                random_state=random_state,
                max_len=max_len
            )

            discretized_data, bin_info, rules = baseline.run()

            zone_output_dir = output_dir / zone_name
            baseline.save_zone_results(
                original_data=data,
                discretized_data=discretized_data,
                bin_info=bin_info,
                rules=rules,
                zone_output_dir=zone_output_dir,
                zone_name=zone_name.replace(" ", "_")
            )

            zone_covered_variables = baseline.extract_variables_from_rules(rules)
            all_covered_variables.update(zone_covered_variables)

            summary_record = baseline.summarize_rules(rules, zone_name)
            per_zone_records.append(summary_record)

            print(f"Zone = {zone_name}")
            print(f"Valid rules = {summary_record['Valid_rules']}")
            print(f"Total rules = {summary_record['Total_rules']}")
            print(f"Max support = {summary_record['Max_support']:.6f}")
            print(f"Mean support = {summary_record['Mean_support']:.6f}")
            print(f"Mean confidence = {summary_record['Mean_confidence']:.6f}")
            print(f"Mean lift = {summary_record['Mean_lift']:.6f}")
            print(f"Mean rule length = {summary_record['Mean_rule_length']:.6f}")
            print(f"Variables covered = {summary_record['Variables_covered_n']}")

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
                "Variables_covered_n": 0
            })

    per_zone_summary = pd.DataFrame(per_zone_records)
    per_zone_summary.to_excel(output_dir / "static_per_zone_summary.xlsx", index=False)

    covered_vars_df = pd.DataFrame({
        "Variable": sorted(list(all_covered_variables))
    })
    covered_vars_df.to_excel(output_dir / "static_all_covered_variables.xlsx", index=False)

    table15_static = build_table15_static_row(
        per_zone_summary=per_zone_summary,
        all_covered_variables=all_covered_variables,
        method_name="Static discretization + Apriori"
    )
    table15_static.to_excel(output_dir / "Table15_static_row.xlsx", index=False)

    with pd.ExcelWriter(output_dir / "static_batch_summary.xlsx", engine="openpyxl") as writer:
        per_zone_summary.to_excel(writer, sheet_name="Per_Zone_Summary", index=False)
        covered_vars_df.to_excel(writer, sheet_name="All_Covered_Variables", index=False)
        table15_static.to_excel(writer, sheet_name="Table15_Static_Row", index=False)

    print("\n" + "=" * 70)
    print("Batch run finished.")
    print(f"Per-zone summary saved to: {output_dir / 'static_per_zone_summary.xlsx'}")
    print(f"Covered variables saved to: {output_dir / 'static_all_covered_variables.xlsx'}")
    print(f"Table 15 static row saved to: {output_dir / 'Table15_static_row.xlsx'}")
    print(f"Combined summary saved to: {output_dir / 'static_batch_summary.xlsx'}")
    print("=" * 70)

    return per_zone_summary, covered_vars_df, table15_static


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

    per_zone_summary, covered_vars_df, table15_static = run_batch_static_baseline(
        input_files=input_files,
        output_dir="static_baseline_batch_output",
        fixed_bins=5,
        integer_unique_threshold=10,
        min_support_thresh=0.1,
        rule_conf_thresh=0.5,
        random_state=42,
        max_len=5
    )

    print("\nPer-zone summary:")
    print(per_zone_summary)

    print("\nAll covered variables:")
    print(covered_vars_df)

    print("\nTable 15 static row:")
    print(table15_static)