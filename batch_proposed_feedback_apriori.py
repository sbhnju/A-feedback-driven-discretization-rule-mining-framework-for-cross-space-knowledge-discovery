import time
from pathlib import Path
import warnings
import random
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from deap import base, creator, tools

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ProposedFeedbackDrivenApriori:
    """
    Proposed feedback-driven discretization + Apriori

    与你当前方法保持一致的核心思想：
    - 先识别常量列 / 整数变量
    - 对连续变量用 KMeans 分箱
    - 用 GA 优化每个变量的分箱数
    - 用规则质量（这里取最高支持度）反向反馈分箱配置
    """

    FITNESS_NAME = "FitnessMaxFeedbackApriori"
    IND_NAME = "IndividualFeedbackApriori"

    def __init__(
        self,
        data: pd.DataFrame,
        min_bins: int = 3,
        max_bins: int = 10,
        integer_unique_threshold: int = 10,
        min_support_thresh: float = 0.1,
        rule_conf_thresh: float = 0.5,
        population_size: int = 30,
        generations: int = 200,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        random_state: int = 42,
        max_len: int = 5,
    ):
        self.raw_data = data.copy()
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.integer_unique_threshold = integer_unique_threshold
        self.min_support_thresh = min_support_thresh
        self.rule_conf_thresh = rule_conf_thresh
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        self.max_len = max_len

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        self.original_data, self.constant_columns, self.integer_like_columns = self.preprocess_data(
            self.raw_data.copy()
        )

        # 只有“非常量且不作为固定整数唯一值处理”的列参与 GA 优化
        self.optimizable_cols = [
            c for c in self.original_data.columns
            if c not in self.constant_columns and c not in self.integer_like_columns
        ]

        self.toolbox = None
        self.eval_cache = {}
        self.history = {
            "generation": [],
            "avg_fitness": [],
            "best_fitness": [],
            "best_config": []
        }

        self.setup_genetic_algorithm()

    @staticmethod
    def is_integer_like(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return False
        return np.all(np.isclose(s, np.round(s)))

    def preprocess_data(self, data: pd.DataFrame):
        constant_cols = []
        integer_like_cols = []

        for col in data.columns:
            non_null = data[col].dropna()
            unique_vals = non_null.nunique()

            if unique_vals <= 1:
                constant_cols.append(col)
            elif self.is_integer_like(non_null):
                integer_like_cols.append(col)

        return data, constant_cols, integer_like_cols

    def setup_genetic_algorithm(self):
        # 避免 DEAP creator 重复定义报错
        if not hasattr(creator, self.FITNESS_NAME):
            creator.create(self.FITNESS_NAME, base.Fitness, weights=(1.0,))
        if not hasattr(creator, self.IND_NAME):
            creator.create(self.IND_NAME, list, fitness=getattr(creator, self.FITNESS_NAME))

        self.toolbox = base.Toolbox()

        # 每个基因 = 一个可优化变量的分箱数
        self.toolbox.register("attr_bins", random.randint, self.min_bins, self.max_bins)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            getattr(creator, self.IND_NAME),
            self.toolbox.attr_bins,
            n=len(self.optimizable_cols),
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual, indpb=0.15)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate_individual(self, individual, indpb=0.15):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(self.min_bins, self.max_bins)
        return (individual,)

    def convert_individual_to_config(self, individual):
        """
        individual: [bins_for_varA, bins_for_varB, ...]
        输出完整配置字典
        """
        config = {}

        # 1) 常量列 -> 1 箱
        for col in self.constant_columns:
            config[col] = 1

        # 2) 整数变量处理
        for col in self.integer_like_columns:
            unique_vals = pd.to_numeric(self.original_data[col], errors="coerce").dropna().nunique()
            if unique_vals <= self.integer_unique_threshold:
                config[col] = int(unique_vals)
            else:
                # 若整数变量唯一值过多，这里仍固定为 min_bins，不进入 GA
                config[col] = self.min_bins

        # 3) 连续变量 / 参与优化变量
        for col, bins in zip(self.optimizable_cols, individual):
            config[col] = int(bins)

        # 防御性补全
        for col in self.original_data.columns:
            if col not in config:
                config[col] = self.min_bins

        return config

    def kmeans_discretize(self, n_bins_settings=None):
        if n_bins_settings is None:
            n_bins_settings = {}
            for col in self.original_data.columns:
                if col in self.constant_columns:
                    n_bins_settings[col] = 1
                elif col in self.integer_like_columns:
                    unique_vals = pd.to_numeric(self.original_data[col], errors="coerce").dropna().nunique()
                    if unique_vals <= self.integer_unique_threshold:
                        n_bins_settings[col] = int(unique_vals)
                    else:
                        n_bins_settings[col] = self.min_bins
                else:
                    n_bins_settings[col] = self.min_bins

        discretized_data = self.original_data.copy()
        bin_info = {}

        # 1) 常量列
        for col in self.constant_columns:
            non_null = self.original_data[col].dropna()
            const_val = non_null.iloc[0] if len(non_null) > 0 else np.nan
            bin_label = f"{col}_Constant"
            discretized_data[col] = bin_label
            bin_info[col] = {
                "strategy": "constant",
                "n_bins": 1,
                "boundaries": [const_val, const_val],
                "value_ranges": {bin_label: (const_val, const_val)}
            }

        # 2) 其余列
        for col in [c for c in self.original_data.columns if c not in self.constant_columns]:
            series = pd.to_numeric(self.original_data[col], errors="coerce")
            unique_vals = series.dropna().nunique()
            k = int(n_bins_settings.get(col, self.min_bins))

            # 2.1 低基数整数变量 -> 按唯一值分箱
            if col in self.integer_like_columns and unique_vals <= self.integer_unique_threshold:
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

            # 2.2 其余变量 -> KMeans 分箱
            discretized_data, bin_info = self.kmeans_discretize_col(
                discretized_data=discretized_data,
                bin_info=bin_info,
                col=col,
                k=k,
                series=series
            )

        return discretized_data, bin_info

    def kmeans_discretize_col(self, discretized_data, bin_info, col, k, series):
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
        k = min(max(1, k), unique_vals)

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
                "strategy": "feedback_kmeans",
                "n_bins": k,
                "boundaries": boundaries,
                "value_ranges": value_ranges
            }
            return discretized_data, bin_info

        except Exception:
            return self.qcut_fallback(discretized_data, bin_info, col, k, series)

    def qcut_fallback(self, discretized_data, bin_info, col, k, series):
        valid = series.dropna()
        unique_vals = valid.nunique()
        k = min(max(1, k), unique_vals)

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
            valid, q=k, labels=labels, retbins=True, duplicates="drop"
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

    def prepare_transactions(self, discretized_data):
        transactions = []
        for _, row in discretized_data.iterrows():
            transaction = []
            for col, val in row.items():
                # 与原代码一致：常量列不进入事务
                if col in self.constant_columns:
                    continue
                if pd.notna(val):
                    transaction.append(str(val))
            transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def mine_association_rules(self, transactions):
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
                # 沿用你原代码中的定义
                rules["lift"] = rules["confidence"] / rules["support"]
                rules["antecedent_len"] = rules["antecedents"].apply(len)
                rules["consequent_len"] = rules["consequents"].apply(len)
                rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

            return rules
        except Exception:
            return pd.DataFrame()

    def rule_quality_score(self, rules):
        """
        与你原方法一致：使用支持度最高的规则作为反馈信号
        """
        if rules is None or rules.empty:
            return 0.0, 0.0, 0.0, 0.0

        max_support_rule = rules.nlargest(1, "support")
        if max_support_rule.empty:
            return 0.0, 0.0, 0.0, 0.0

        max_support = float(max_support_rule["support"].values[0])
        confidence = float(max_support_rule["confidence"].values[0]) if "confidence" in max_support_rule else 0.0
        lift = float(max_support_rule["lift"].values[0]) if "lift" in max_support_rule else 0.0
        antecedent_len = float(max_support_rule["antecedent_len"].values[0]) if "antecedent_len" in max_support_rule else 0.0

        return max_support, max_support, confidence, lift

    def evaluate_individual(self, individual):
        """
        GA 适应度：支持度最高规则的 support
        """
        key = tuple(individual)
        if key in self.eval_cache:
            return (self.eval_cache[key],)

        try:
            n_bins_settings = self.convert_individual_to_config(individual)
            discretized_data, _ = self.kmeans_discretize(n_bins_settings)
            transactions = self.prepare_transactions(discretized_data)
            rules = self.mine_association_rules(transactions)

            quality_score, max_support, _, _ = self.rule_quality_score(rules)
            self.eval_cache[key] = quality_score
            return (quality_score,)
        except Exception:
            self.eval_cache[key] = 0.0
            return (0.0,)

    def run_optimization(self):
        pop = self.toolbox.population(n=self.population_size)

        # 初始评估
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 变异
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估无效个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            pop[:] = offspring

            fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
            avg_fitness = float(np.mean(fits)) if fits else 0.0
            best_fitness = float(np.max(fits)) if fits else 0.0
            best_ind = tools.selBest(pop, k=1)[0]
            best_config = self.convert_individual_to_config(best_ind)

            self.history["generation"].append(gen + 1)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_fitness"].append(best_fitness)
            self.history["best_config"].append(best_config)

        best_ind = tools.selBest(pop, k=1)[0]
        best_fitness = float(best_ind.fitness.values[0]) if best_ind.fitness.valid else 0.0
        best_config = self.convert_individual_to_config(best_ind)

        return best_config, best_fitness

    def apply_best_config(self, best_config):
        discretized_data, bin_info = self.kmeans_discretize(best_config)
        return discretized_data, bin_info

    def summarize_rules(self, rules: pd.DataFrame, zone_name: str, best_fitness: float, runtime_sec: float):
        if rules is None or rules.empty:
            return {
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Max_support": 0.0,
                "Best_fitness": best_fitness,
                "Run_time_sec": runtime_sec
            }

        return {
            "Zone": zone_name,
            "Valid_rules": 1,
            "Total_rules": int(len(rules)),
            "Mean_support": float(rules["support"].mean()),
            "Mean_confidence": float(rules["confidence"].mean()),
            "Mean_lift": float(rules["lift"].mean()),
            "Mean_rule_length": float(rules["rule_len"].mean()),
            "Max_support": float(rules["support"].max()),
            "Best_fitness": best_fitness,
            "Run_time_sec": runtime_sec
        }

    def save_zone_results(
        self,
        original_data: pd.DataFrame,
        discretized_data: pd.DataFrame,
        bin_info: dict,
        rules: pd.DataFrame,
        best_config: dict,
        zone_output_dir: Path,
        zone_name: str
    ):
        zone_output_dir.mkdir(parents=True, exist_ok=True)

        # 1) 原始 + 离散化结果
        combined_data = pd.concat(
            [original_data, discretized_data.add_prefix("Disc_")],
            axis=1
        )
        combined_data.to_excel(
            zone_output_dir / f"{zone_name}_proposed_discretized_data.xlsx",
            index=False
        )

        # 2) 最佳配置
        config_df = pd.DataFrame(
            [{"Variable": k, "Bins": v} for k, v in best_config.items()]
        )
        config_df.to_excel(
            zone_output_dir / f"{zone_name}_proposed_best_bins_config.xlsx",
            index=False
        )

        # 3) 分箱细节
        bin_rows = []
        for col, info in bin_info.items():
            bin_rows.append({
                "Variable": col,
                "Strategy": info.get("strategy", ""),
                "Bins": info.get("n_bins", ""),
                "Boundaries": str(info.get("boundaries", "")),
                "Value_Ranges": str(info.get("value_ranges", ""))
            })
        pd.DataFrame(bin_rows).to_excel(
            zone_output_dir / f"{zone_name}_proposed_bin_details.xlsx",
            index=False
        )

        # 4) 规则
        if rules is not None and not rules.empty:
            rules_to_save = rules.copy()
            rules_to_save["antecedents"] = rules_to_save["antecedents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            rules_to_save["consequents"] = rules_to_save["consequents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            rules_to_save.to_excel(
                zone_output_dir / f"{zone_name}_proposed_association_rules.xlsx",
                index=False
            )

            high_support_rules = rules_to_save[rules_to_save["support"] > 0.5]
            high_support_rules.to_excel(
                zone_output_dir / f"{zone_name}_proposed_high_support_rules.xlsx",
                index=False
            )


def load_zone_data(file_path: Path, n_vars: int = 12) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine="openpyxl")

    if len(df.columns) < n_vars:
        num_cols = len(df.columns)
    else:
        num_cols = n_vars

    data = df.iloc[:, :num_cols].copy()
    data.columns = [f"Var{i+1}" for i in range(num_cols)]
    return data


def build_table15_proposed_row(per_zone_summary: pd.DataFrame,
                               method_name: str = "Proposed method") -> pd.DataFrame:
    """
    生成可直接用于 Table 15 的一行
    与 static / NiaARM 保持同口径：按各 zone 规则数加权平均
    """
    valid_zone_count = int(per_zone_summary["Valid_rules"].sum())
    total_rules = int(per_zone_summary["Total_rules"].sum())

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

    return pd.DataFrame([{
        "Method": method_name,
        "Occupied zones with valid rules, n": valid_zone_count,
        "Total rules, n": total_rules,
        "Mean support": mean_support,
        "Mean confidence": mean_confidence,
        "Mean lift": mean_lift,
        "Mean rule length": mean_rule_length
    }])


def run_batch_proposed_feedback(
    input_files,
    output_dir="proposed_feedback_batch_output",
    min_bins=3,
    max_bins=10,
    integer_unique_threshold=10,
    min_support_thresh=0.1,
    rule_conf_thresh=0.5,
    population_size=30,
    generations=200,
    crossover_prob=0.7,
    mutation_prob=0.3,
    random_state=42,
    max_len=5
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_zone_records = []

    for file_name in input_files:
        file_path = Path(file_name)
        zone_name = file_path.stem

        print("\n" + "=" * 72)
        print(f"Processing proposed feedback-driven method for: {file_name}")
        print("=" * 72)

        if not file_path.exists():
            print(f"文件不存在，跳过: {file_name}")
            per_zone_records.append({
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Max_support": 0.0,
                "Best_fitness": 0.0,
                "Run_time_sec": np.nan
            })
            continue

        try:
            t0 = time.time()

            data = load_zone_data(file_path, n_vars=12)
            print(f"加载成功: {file_name} | shape = {data.shape}")

            optimizer = ProposedFeedbackDrivenApriori(
                data=data,
                min_bins=min_bins,
                max_bins=max_bins,
                integer_unique_threshold=integer_unique_threshold,
                min_support_thresh=min_support_thresh,
                rule_conf_thresh=rule_conf_thresh,
                population_size=population_size,
                generations=generations,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                random_state=random_state,
                max_len=max_len
            )

            best_config, best_fitness = optimizer.run_optimization()
            discretized_data, bin_info = optimizer.apply_best_config(best_config)
            transactions = optimizer.prepare_transactions(discretized_data)
            rules = optimizer.mine_association_rules(transactions)

            elapsed = time.time() - t0

            zone_output_dir = output_dir / zone_name
            optimizer.save_zone_results(
                original_data=data,
                discretized_data=discretized_data,
                bin_info=bin_info,
                rules=rules,
                best_config=best_config,
                zone_output_dir=zone_output_dir,
                zone_name=zone_name.replace(" ", "_")
            )

            summary_record = optimizer.summarize_rules(
                rules=rules,
                zone_name=zone_name,
                best_fitness=best_fitness,
                runtime_sec=elapsed
            )
            per_zone_records.append(summary_record)

            print(f"Zone = {zone_name}")
            print(f"Best fitness = {best_fitness:.6f}")
            print(f"Valid rules = {summary_record['Valid_rules']}")
            print(f"Total rules = {summary_record['Total_rules']}")
            print(f"Mean support = {summary_record['Mean_support']:.6f}")
            print(f"Mean confidence = {summary_record['Mean_confidence']:.6f}")
            print(f"Mean lift = {summary_record['Mean_lift']:.6f}")
            print(f"Mean rule length = {summary_record['Mean_rule_length']:.6f}")
            print(f"Run time (s) = {summary_record['Run_time_sec']:.2f}")

        except Exception as e:
            print(f"处理失败: {file_name} | Error: {e}")
            per_zone_records.append({
                "Zone": zone_name,
                "Valid_rules": 0,
                "Total_rules": 0,
                "Mean_support": 0.0,
                "Mean_confidence": 0.0,
                "Mean_lift": 0.0,
                "Mean_rule_length": 0.0,
                "Max_support": 0.0,
                "Best_fitness": 0.0,
                "Run_time_sec": np.nan
            })

    # 1) 每个 zone 的统计汇总
    per_zone_summary = pd.DataFrame(per_zone_records)
    per_zone_summary.to_excel(output_dir / "proposed_per_zone_summary.xlsx", index=False)

    # 2) 生成可直接用于 Table 15 的 proposed 一行
    table15_proposed = build_table15_proposed_row(
        per_zone_summary=per_zone_summary,
        method_name="Proposed method"
    )
    table15_proposed.to_excel(output_dir / "Table15_proposed_row.xlsx", index=False)

    # 3) 输出总览
    with pd.ExcelWriter(output_dir / "proposed_batch_summary.xlsx", engine="openpyxl") as writer:
        per_zone_summary.to_excel(writer, sheet_name="Per_Zone_Summary", index=False)
        table15_proposed.to_excel(writer, sheet_name="Table15_Proposed_Row", index=False)

    print("\n" + "=" * 72)
    print("Proposed feedback-driven batch run finished.")
    print(f"Per-zone summary saved to: {output_dir / 'proposed_per_zone_summary.xlsx'}")
    print(f"Table 15 proposed row saved to: {output_dir / 'Table15_proposed_row.xlsx'}")
    print(f"Combined summary saved to: {output_dir / 'proposed_batch_summary.xlsx'}")
    print("=" * 72)

    return per_zone_summary, table15_proposed


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

    per_zone_summary, table15_proposed = run_batch_proposed_feedback(
        input_files=input_files,
        output_dir="proposed_feedback_batch_output",
        min_bins=3,
        max_bins=10,
        integer_unique_threshold=10,
        min_support_thresh=0.1,
        rule_conf_thresh=0.5,
        population_size=30,
        generations=200,      # 若想先试跑，可先改成 30 或 50
        crossover_prob=0.7,
        mutation_prob=0.3,
        random_state=42,
        max_len=5
    )

    print("\nPer-zone summary:")
    print(per_zone_summary)

    print("\nTable 15 proposed row:")
    print(table15_proposed)