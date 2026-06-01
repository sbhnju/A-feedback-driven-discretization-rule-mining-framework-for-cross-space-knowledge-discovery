import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import random
from deap import base, creator, tools, algorithms

# 设置Matplotlib全局使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果系统有宋体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

warnings.filterwarnings('ignore', category=RuntimeWarning)


class GeneticDiscretizationOptimizer:
    def __init__(self, data, min_bins=3, max_bins=8,
                 min_support_thresh=0.1, rule_conf_thresh=0.6,
                 population_size=20, generations=10, crossover_prob=0.7, mutation_prob=0.2):
        """
        遗传算法优化器初始化
        """
        # 预处理：识别唯一值变量和整数变量
        self.original_data, self.constant_columns, self.integer_columns = self.preprocess_data(data.copy())
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.min_support_thresh = min_support_thresh
        self.rule_conf_thresh = rule_conf_thresh

        # 遗传算法参数
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # 设置遗传算法
        self.setup_genetic_algorithm()

        # 保存优化历史
        self.history = {
            'generation': [],
            'avg_fitness': [],
            'best_fitness': [],
            'best_config': []
        }

    def preprocess_data(self, data):
        """预处理数据：识别唯一值变量和整数变量，返回处理后的数据和列列表"""
        constant_cols = []  # 唯一值变量
        int_cols = []  # 整数变量（非唯一值）

        for col in data.columns:
            unique_vals = data[col].nunique()

            # 记录唯一值变量
            if unique_vals == 1:
                constant_cols.append(col)
                print(f"信息: 列 '{col}' 是唯一值变量（将分为1箱）")

            # 记录整数型变量（非唯一值）
            elif data[col].dtype.kind in 'iu':
                int_cols.append(col)
                print(f"信息: 列 '{col}' 是整数变量")

        # 注意：不再移除唯一值变量
        return data, constant_cols, int_cols

    def setup_genetic_algorithm(self):
        """设置遗传算法环境"""
        # 创建适应度和个体类型
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # 定义个体的属性: (列索引, 分箱数)
        # 仅优化非整数变量（不包括常量列和整数变量）
        self.optimizable_cols = [col for col in self.original_data.columns
                                 if col not in self.integer_columns and col not in self.constant_columns]

        print(f"共有 {len(self.optimizable_cols)} 个变量参与优化")
        print(f"常量列 ({len(self.constant_columns)}): {', '.join(self.constant_columns)}")
        print(f"整数列 ({len(self.integer_columns)}): {', '.join(self.integer_columns)}")

        # 注册生成基因的函数
        def gene_generator():
            # 从可优化列中选择一个列及其分箱数
            col_name = random.choice(self.optimizable_cols)
            col_idx = self.optimizable_cols.index(col_name)
            bins = random.randint(self.min_bins, self.max_bins)
            return (col_idx, bins)

        # 注册生成个体的函数
        self.toolbox.register("attr_gene", gene_generator)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_gene, n=len(self.optimizable_cols))

        # 注册生成种群的函数
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 注册评估函数
        self.toolbox.register("evaluate", self.evaluate_individual)

        # 注册遗传算子
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate_individual(self, individual, indpb):
        """变异操作"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # 变异分箱数
                current_idx, current_bins = individual[i]
                new_bins = random.randint(self.min_bins, self.max_bins)
                individual[i] = (current_idx, new_bins)

        return individual,

    def kmeans_discretize(self, n_bins_settings=None):
        """
        使用K-means聚类进行离散化
        """
        # 1. 为所有列设置默认分箱数（常量列固定为1箱）
        if n_bins_settings is None:
            n_bins_settings = {}
            for col in self.original_data.columns:
                if col in self.constant_columns:  # 常量列
                    n_bins_settings[col] = 1
                elif col in self.integer_columns:  # 整数变量
                    unique_vals = self.original_data[col].nunique()
                    if unique_vals <= self.max_bins:
                        n_bins_settings[col] = unique_vals
                    else:
                        n_bins_settings[col] = self.min_bins
                else:  # 其他变量
                    n_bins_settings[col] = self.min_bins

        discretized_data = self.original_data.copy()
        bin_info = {}

        # 2. 处理常量列
        for col in self.constant_columns:
            constant_value = self.original_data[col].iloc[0]  # 获取唯一值
            bin_label = f"{col}_Constant"
            discretized_data[col] = bin_label
            bin_info[col] = {
                'boundaries': [constant_value, constant_value],
                'value_ranges': {bin_label: (constant_value, constant_value)}
            }
            print(f"列 '{col}' 是唯一值变量，设置为1个分箱: {bin_label}")

        # 3. 处理非常量列
        if not self.original_data.empty and len(
                [c for c in self.original_data.columns if c not in self.constant_columns]) > 0:
            # 仅对非常量列进行标准化
            non_constant_cols = [c for c in self.original_data.columns if c not in self.constant_columns]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.original_data[non_constant_cols])
        else:
            scaled_data = None

        # 处理非常量列
        for i, col in enumerate([c for c in self.original_data.columns if c not in self.constant_columns]):
            k = n_bins_settings[col]

            # 对于整数变量（非常量）
            if col in self.integer_columns and self.original_data[col].nunique() <= self.max_bins:
                try:
                    # 获取唯一值并排序
                    unique_vals = sorted(self.original_data[col].unique())

                    # 创建映射字典：数值 -> 类别标签
                    val_to_label = {val: f"{col}_Int{idx}" for idx, val in enumerate(unique_vals)}

                    # 应用离散化
                    discretized_data[col] = self.original_data[col].map(val_to_label)

                    # 为整数变量添加边界信息
                    bin_info[col] = {
                        'boundaries': unique_vals,
                        'value_ranges': {val_to_label[val]: (val, val) for val in unique_vals}
                    }
                    print(f"列 '{col}' 使用整数分箱，分箱数: {len(unique_vals)}")
                except Exception as e:
                    print(f"列 '{col}' 整数分箱失败: {str(e)}")
                    # 使用K-means作为备选
                    col_idx = non_constant_cols.index(col)
                    discretized_data, bin_info = self.kmeans_discretize_col(
                        discretized_data, bin_info, col, k, scaler, col_idx,
                        scaled_data[:, col_idx] if scaled_data is not None else None)
                continue

            # 对于其他变量使用K-means聚类
            if scaled_data is not None:
                col_idx = non_constant_cols.index(col)
                discretized_data, bin_info = self.kmeans_discretize_col(
                    discretized_data, bin_info, col, k, scaler, col_idx, scaled_data[:, col_idx])
            else:
                # 如果数据为空，直接赋值
                discretized_data[col] = f"{col}_NoData"
                bin_info[col] = {
                    'boundaries': [0, 0],
                    'value_ranges': {f"{col}_NoData": (0, 0)}
                }

        return discretized_data, bin_info

    def kmeans_discretize_col(self, discretized_data, bin_info, col, k, scaler, i, scaled_data):
        """使用K-means进行离散化（单列）- 修正版本（使用实际聚类值域）"""
        try:
            # 处理唯一值少于分箱数的情况
            unique_vals = len(np.unique(self.original_data[col]))
            if unique_vals <= k:
                k = max(2, min(unique_vals, k))

            col_data = scaled_data.reshape(-1, 1) if scaled_data is not None else self.original_data[col].values.reshape(-1, 1)

            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(col_data)

            # 为每个聚类创建临时分组
            clusters = [[] for _ in range(k)]
            original_values = self.original_data[col].values

            for idx, label in enumerate(labels):
                clusters[label].append(original_values[idx])

            # 计算每个聚类的实际最小值和最大值
            cluster_boundaries = {}
            for label in range(k):
                if clusters[label]:  # 确保聚类不为空
                    cluster_min = np.min(clusters[label])
                    cluster_max = np.max(clusters[label])
                    cluster_boundaries[label] = (cluster_min, cluster_max)
                else:
                    # 处理空聚类
                    cluster_boundaries[label] = (np.nan, np.nan)

            # 按最小值对聚类排序，并创建映射关系
            sorted_clusters = sorted(cluster_boundaries.items(), key=lambda x: x[1][0])
            sorted_labels = [item[0] for item in sorted_clusters]  # 原始聚类索引排序

            # 创建分箱边界和实际值域
            boundaries = []
            actual_value_ranges = {}
            bin_label_map = {}  # 映射原始聚类索引 -> 排序后标签

            # 排序后的第一个分箱的左边界
            boundaries.append(sorted_clusters[0][1][0])

            # 创建映射关系和分箱标签
            for new_idx, (orig_label, (min_val, max_val)) in enumerate(sorted_clusters):
                # 创建按值域排序的分箱标签
                bin_label = f"{col}_Bin{new_idx}"
                bin_label_map[orig_label] = bin_label  # 映射原始聚类索引到新标签
                actual_value_ranges[bin_label] = (min_val, max_val)
                boundaries.append(max_val)

            # 保存实际值域用于可视化
            bin_info[col] = {
                'boundaries': boundaries,
                'value_ranges': actual_value_ranges,
                'bin_label_map': bin_label_map  # 保存映射关系
            }

            # 为每个数据点分配分箱标签（使用映射关系）
            bin_assignment = [bin_label_map[label] for label in labels]

            # 应用分箱
            discretized_data[col] = bin_assignment

            #print(f"列 '{col}' K-means分箱成功，分箱数: {k}")
            return discretized_data, bin_info

        except Exception as e:
            print(f"列 '{col}' K-means分箱失败: {str(e)}")
            # 尝试等频分箱
            try:
                unique_vals = len(np.unique(self.original_data[col]))
                k = min(k, unique_vals)  # 确保分箱数不超过唯一值数量
                discretized, bins = pd.qcut(self.original_data[col], q=k,
                                            labels=[f"{col}_Bin{i}" for i in range(k)],
                                            retbins=True, duplicates='drop')
                discretized_data[col] = discretized

                # 记录实际值域
                value_ranges = {}
                for i in range(k):
                    bin_label = f"{col}_Bin{i}"
                    bin_values = self.original_data[col][discretized_data[col] == bin_label]
                    if len(bin_values) > 0:
                        min_val = np.min(bin_values)
                        max_val = np.max(bin_values)
                        value_ranges[bin_label] = (min_val, max_val)

                bin_info[col] = {
                    'boundaries': bins.tolist(),
                    'value_ranges': value_ranges
                }
                print(f"列 '{col}' 使用等频分箱成功")
                return discretized_data, bin_info

            except Exception as e2:
                print(f"列 '{col}' 等频分箱失败: {str(e2)}")
                # 尝试等宽分箱
                try:
                    unique_vals = len(np.unique(self.original_data[col]))
                    k = min(k, unique_vals)  # 确保分箱数不超过唯一值数量
                    discretized, bins = pd.cut(self.original_data[col], bins=k,
                                               labels=[f"{col}_Bin{i}" for i in range(k)],
                                               retbins=True, duplicates='drop')
                    discretized_data[col] = discretized

                    # 记录实际值域
                    value_ranges = {}
                    for i in range(k):
                        bin_label = f"{col}_Bin{i}"
                        bin_values = self.original_data[col][discretized_data[col] == bin_label]
                        if len(bin_values) > 0:
                            min_val = np.min(bin_values)
                            max_val = np.max(bin_values)
                            value_ranges[bin_label] = (min_val, max_val)

                    bin_info[col] = {
                        'boundaries': bins.tolist(),
                        'value_ranges': value_ranges
                    }
                    print(f"列 '{col}' 使用等宽分箱成功")
                    return discretized_data, bin_info

                except Exception as e3:
                    print(f"列 '{col}' 所有分箱方法均失败: {str(e3)}")
                    # 所有方法都失败时使用单值处理
                    discretized_data[col] = f"{col}_SingleValue"

                    # 尝试计算实际值域
                    try:
                        bin_info[col] = {
                            'boundaries': [self.original_data[col].min(), self.original_data[col].max()],
                            'value_ranges': {f"{col}_SingleValue": (
                                self.original_data[col].min(),
                                self.original_data[col].max()
                            )}
                        }
                    except:
                        bin_info[col] = {
                            'boundaries': [0, 0],
                            'value_ranges': {f"{col}_SingleValue": (0, 0)}
                        }

                    return discretized_data, bin_info

    def prepare_transactions(self, discretized_data):
        """
        准备事务数据用于关联规则挖掘
        """
        transactions = []
        for _, row in discretized_data.iterrows():
            transaction = []
            for col, val in row.items():
                # 跳过常量列（唯一值变量）
                if col in self.constant_columns:
                    continue

                if pd.notna(val):
                    # 添加到事务
                    transaction.append(str(val))
            transactions.append(transaction)

        # 使用TransactionEncoder转换数据
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def mine_association_rules(self, transactions):
        """
        挖掘关联规则
        """
        # 如果没有非常量列，则跳过关联规则挖掘
        if len([c for c in self.original_data.columns if c not in self.constant_columns]) == 0:
            print("所有变量都是常量列，跳过关联规则挖掘")
            return pd.DataFrame()

        # 设置合理的支持度阈值
        if transactions.empty:
            print("警告: 没有有效事务数据用于关联规则挖掘")
            return pd.DataFrame()

        min_support = max(self.min_support_thresh, 50 / len(transactions))

        try:
            # 过滤空事务
            if transactions.empty or transactions.shape[1] == 0:
                print("警告: 没有有效事务数据用于关联规则挖掘")
                return pd.DataFrame()

            frequent_itemsets = apriori(transactions, min_support=min_support,
                                        use_colnames=True, max_len=5)

            if frequent_itemsets.empty:
                print("未找到频繁项集")
                return pd.DataFrame()

            rules = association_rules(frequent_itemsets, metric="confidence",
                                      min_threshold=self.rule_conf_thresh)

            # 添加额外指标
            if not rules.empty:
                rules['lift'] = rules['confidence'] / rules['support']
                rules['antecedent_len'] = rules['antecedents'].apply(len)
                rules['consequent_len'] = rules['consequents'].apply(len)

            return rules
        except Exception as e:
            print(f"关联规则挖掘失败: {e}")
            return pd.DataFrame()

    def rule_quality_score(self, rules):
        """
        计算关联规则质量得分（仅考虑支持度最高的规则）
        """
        if rules.empty or len(rules) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 找到支持度最高的规则
        max_support_rule = rules.nlargest(1, 'support')

        if max_support_rule.empty:
            return 0.0, 0.0, 0.0, 0.0

        # 获取最高支持度
        max_support = max_support_rule['support'].values[0]

        # 其他指标（可选）
        confidence = max_support_rule['confidence'].values[0] if 'confidence' in max_support_rule else 0.0
        lift = max_support_rule['lift'].values[0] if 'lift' in max_support_rule else 0.0
        antecedent_len = max_support_rule['antecedent_len'].values[0] if 'antecedent_len' in max_support_rule else 0.0

        # 使用最高支持度作为质量评分
        quality_score = max_support

        return quality_score, max_support, confidence, lift

    def evaluate_individual(self, individual):
        """
        评估个体的适应度（规则质量得分）
        """
        try:
            # 将个体编码转换为分箱配置
            n_bins_settings = self.convert_individual_to_config(individual)

            # 执行离散化
            discretized_data, _ = self.kmeans_discretize(n_bins_settings)

            # 准备事务数据
            transactions = self.prepare_transactions(discretized_data)

            # 挖掘关联规则
            rules = self.mine_association_rules(transactions)

            # 计算规则质量得分（仅使用最高支持度）
            quality_score, max_support, _, _ = self.rule_quality_score(rules)

            # 输出调试信息
            print(f"个体配置支持度最高的规则得分: {max_support:.4f}")

            return quality_score,
        except Exception as e:
            print(f"评估个体时出错: {e}")
            return 0.0,  # 返回最低适应度

    def convert_individual_to_config(self, individual):
        """
        将遗传算法的个体转换为分箱配置字典
        """
        config = {}

        # 常量列固定为1箱
        for col in self.constant_columns:
            config[col] = 1

        # 整数变量的处理（非常量）
        for col in [c for c in self.integer_columns if c not in self.constant_columns]:
            if col in self.original_data.columns:  # 确保列存在
                unique_vals = self.original_data[col].nunique()
                if unique_vals <= self.max_bins:
                    config[col] = unique_vals
                else:
                    config[col] = self.min_bins  # 默认值

        # 非整数变量的处理（使用遗传算法优化的值）
        for col_idx, bins in individual:
            if col_idx < len(self.optimizable_cols):
                col_name = self.optimizable_cols[col_idx]
                if col_name in self.original_data.columns:  # 确保列存在
                    config[col_name] = bins

        # 对于没有配置的列使用默认值
        for col in self.original_data.columns:
            if col not in config:
                config[col] = self.min_bins

        return config

    def run_optimization(self):
        """
        执行遗传算法优化
        """
        print("启动遗传算法优化...")
        print(f"种群大小: {self.population_size}, 代数: {self.generations}")
        print(f"可优化变量数: {len(self.optimizable_cols)}")

        # 生成初始种群
        pop = self.toolbox.population(n=self.population_size)

        # 评估初始种群
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 运行遗传算法
        for gen in range(self.generations):
            print(f"\n{'=' * 40}")
            print(f"代数: {gen + 1}/{self.generations}")

            # 选择下一代
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # 应用交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估新的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # 更新种群
            pop[:] = offspring

            # 收集统计信息
            fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
            if fits:
                avg_fitness = sum(fits) / len(fits)
                best_fitness = max(fits)
            else:
                avg_fitness = 0.0
                best_fitness = 0.0

            self.history['generation'].append(gen + 1)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_fitness'].append(best_fitness)

            print(f"平均适应度: {avg_fitness:.4f}")
            print(f"最佳适应度: {best_fitness:.4f}")

        # 选择最佳个体
        best_ind = tools.selBest(pop, k=1)[0]
        best_fitness = best_ind.fitness.values[0] if best_ind.fitness.valid else 0.0
        best_config = self.convert_individual_to_config(best_ind)
        self.history['best_config'].append(best_config)

        print(f"\n优化完成! 最佳适应度: {best_fitness:.4f}")
        print("最佳分箱配置:")
        for col, bins in best_config.items():
            print(f"- {col}: {bins} bins")

        return best_config, best_fitness

    def visualize_optimization(self):
        """
        可视化遗传算法优化过程
        """
        if len(self.history['generation']) == 0:
            print("没有优化历史数据可显示")
            return

        plt.figure(figsize=(12, 6))

        plt.plot(self.history['generation'], self.history['avg_fitness'], 'b-', label="平均适应度")
        plt.plot(self.history['generation'], self.history['best_fitness'], 'r-', label="最佳适应度")

        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('遗传算法优化过程')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('genetic_optimization.png', dpi=300)
        plt.show()

    def apply_best_config(self):
        """应用最佳分箱配置并返回离散化结果"""
        if not self.history.get('best_config'):
            print("没有找到最佳配置，请先运行优化")
            return None, None

        best_config = self.history['best_config'][0]
        discretized_data, bin_info = self.kmeans_discretize(best_config)
        return discretized_data, bin_info


def visualize_binning(original_data, discretized_data, bin_info):
    """
    可视化分箱结果并输出边界信息
    """
    if original_data is None or original_data.empty:
        print("没有数据可供可视化")
        return

    n_cols = min(4, len(original_data.columns))
    n_rows = int(np.ceil(len(original_data.columns) / n_cols))

    # 计算所需的图形高度
    fig_height = 5 * n_rows  # 基础高度
    extra_height = min(1, n_rows) * 1.2  # 为图例添加额外高度
    total_height = fig_height + extra_height

    # 使用预定义的颜色循环
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, total_height), squeeze=False)

    # 加大水平间距防止重叠
    fig.subplots_adjust(hspace=0.7, wspace=0.3, bottom=0.15)  # 增加垂直间距

    # 展平坐标轴数组以便遍历
    axes_flat = axes.flatten()

    # 创建字典存储每个分箱的有序百分比信息
    # 结构: {列名: {'percentages': {分箱标签: 百分比}, 'labels': [有序的分箱标签]}}
    bin_percentages = {}

    for col in discretized_data.columns:
        if col in bin_info and 'value_ranges' in bin_info[col]:
            value_ranges = bin_info[col]['value_ranges']

            # 按值域范围排序分箱（从小到大）
            sorted_ranges = sorted(value_ranges.items(), key=lambda x: x[1][0])
            sorted_labels = [label for label, _ in sorted_ranges]

            # 计算每个分箱的原始频率
            col_dist = discretized_data[col].value_counts(normalize=True).to_dict()

            # 创建有序百分比字典
            ordered_percent = {}
            for label in sorted_labels:
                if label in col_dist:
                    ordered_percent[label] = col_dist[label] * 100  # 转换为百分比
                else:
                    ordered_percent[label] = 0.0

            # 保存有序的分箱标签和百分比
            bin_percentages[col] = {
                'labels': sorted_labels,
                'percentages': ordered_percent
            }

    # 绘制每个子图
    for i, col in enumerate(original_data.columns):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        # 绘制直方图 - 使用灰色
        values = original_data[col].dropna()
        if len(values) > 0:
            # 计算合适的直方图分箱数
            hist_bins = min(40, max(10, int(np.sqrt(len(values)))))
            hist_bins = max(5, hist_bins)  # 确保至少5个分箱

            # 使用灰色直方图
            hist, bins, patches = ax.hist(values, bins=hist_bins,
                                          alpha=0.7, color='Silver', edgecolor='Gray', linewidth=0.5)

            # 设置标题和标签
            ax.set_title(f'{col}', fontsize=12)
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(alpha=0.2)

            # 刻度标签两位小数
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            # 如果没有分箱信息，跳过
            if col not in bin_info or 'value_ranges' not in bin_info[col]:
                continue

            # 如果是常量列，特殊处理
            if col in bin_info and len(bin_info[col]['value_ranges']) == 1:
                # 只有一个分箱的情况
                bin_label, (min_val, max_val) = list(bin_info[col]['value_ranges'].items())[0]

                # 计算百分比
                col_total = len(original_data[col])
                col_count = (original_data[col] == min_val).sum()
                percent = (col_count / col_total) * 100

                # 创建图例句柄和标签
                legend_handles = []
                legend_labels = []

                # 选择颜色
                color = colors[0]  # 第一个颜色

                # 添加背景区域（整个范围）
                # 对于常量值，背景区域应该只覆盖该值所在的位置
                ax.axvspan(min_val, max_val, alpha=0.2, color=color, zorder=0)

                # 在常量值处添加垂直线
                ax.axvline(x=min_val, color=color, linewidth=1, linestyle='-', zorder=1)

                # 创建图例标签 - 使用与其他变量相同的格式
                legend_label = f"{bin_label}: [{min_val:.2f}, {max_val:.2f}] ({percent:.2f}%)"
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.2, ec=color))
                legend_labels.append(legend_label)

                # 创建图例
                legend = ax.legend(legend_handles, legend_labels,
                                   loc='upper center',
                                   ncol=1,
                                   fontsize=8,
                                   frameon=True,
                                   bbox_to_anchor=(0.5, -0.15),
                                   borderaxespad=0,
                                   framealpha=0.8,
                                   borderpad=0,
                                   labelspacing=0.1,
                                   handlelength=1.0)

                # 设置图例框样式
                frame = legend.get_frame()
                frame.set_linewidth(0.1)
                frame.set_edgecolor('white')

                continue

            value_ranges = bin_info[col]['value_ranges']
            sorted_ranges = sorted(value_ranges.items(), key=lambda x: x[1][0])

            # 创建图例句柄和标签
            legend_handles = []
            legend_labels = []

            # 确保背景区域不遮盖直方图
            ax.set_autoscaley_on(True)

            # 获取该列的分箱占比信息（修复了百分比获取方式）
            if col in bin_percentages:
                col_pct_info = bin_percentages[col]
                percentages_dict = col_pct_info['percentages']
            else:
                percentages_dict = {label: 0.0 for label, _ in sorted_ranges}

            for idx, (bin_label, (min_val, max_val)) in enumerate(sorted_ranges):
                # 选择颜色
                color = colors[idx % len(colors)]

                # 添加背景区域
                ax.axvspan(min_val, max_val, alpha=0.2, color=color, zorder=0)

                # 添加边界线
                ax.axvline(x=min_val, color=color, linewidth=1, linestyle='-', zorder=1)
                ax.axvline(x=max_val, color=color, linewidth=1, linestyle='-', zorder=1)

                # 获取该分箱的百分比（确保正确获取）
                percent = percentages_dict.get(bin_label, 0.0)
                legend_label = f"{bin_label}: [{min_val:.2f}, {max_val:.2f}] ({percent:.2f}%)"

                legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.2, ec=color))
                legend_labels.append(legend_label)

            # 创建图例放在横坐标轴名称下方，紧挨着，且居中
            if legend_handles and legend_labels:
                # 单列图例
                ncol_legend = 1

                # 创建图例，位置为横坐标轴正下方
                legend = ax.legend(legend_handles, legend_labels,
                                   loc='upper center',
                                   ncol=ncol_legend,
                                   fontsize=8,
                                   frameon=True,
                                   bbox_to_anchor=(0.5, -0.15),  # 横坐标轴正下方
                                   borderaxespad=0,  # 取消边框间距
                                   framealpha=0.8,
                                   borderpad=0,  # 取消图例内边距
                                   labelspacing=0.1,  # 最小化标签间距
                                   handlelength=1.0)  # 图例句柄长度

                # 设置图例框样式
                frame = legend.get_frame()
                frame.set_linewidth(0.1)
                frame.set_edgecolor('white')

                # 确保横坐标轴标签下方没有额外空间
                ax.xaxis.labelpad = 0.5  # 将横坐标轴标签上移，与图例紧密接触

        else:
            # 如果列中没有数据
            ax.text(0.5, 0.5, f"No data in column {col}",
                    ha='center', va='center', transform=ax.transAxes)

    # 隐藏多余的坐标轴
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.savefig('binning_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存

    print("\n分箱可视化已保存为 binning_visualization.png")

    # 打印汇总的分箱信息（值域和分布）
    print("\n\n每个变量的分箱汇总（值域和分布）:")
    for col in bin_info:
        if 'value_ranges' not in bin_info[col]:
            continue

        # 如果是常量列 - 修改这里的显示格式
        if len(bin_info[col]['value_ranges']) == 1:
            bin_label, (min_val, max_val) = list(bin_info[col]['value_ranges'].items())[0]

            # 计算常量列的分箱占比（100%）
            col_total = len(original_data[col])
            col_count = (original_data[col] == min_val).sum()
            percent = (col_count / col_total) * 100

            print(f"\n{col}分箱汇总:")
            # 使用与其他变量相同的格式
            print(f"  {bin_label}: 值域 [{min_val:.2f}, {max_val:.2f}], 占比 {percent:.2f}%")
            continue

        # 获取分箱值域
        value_ranges = bin_info[col]['value_ranges']
        sorted_ranges = sorted(value_ranges.items(), key=lambda x: x[1][0])

        # 获取该列的分箱占比信息
        if col in bin_percentages:
            col_pct_info = bin_percentages[col]
            percentages_dict = col_pct_info['percentages']
        else:
            percentages_dict = {}

        print(f"\n{col}分箱汇总:")

        # 确保使用正确的顺序输出
        for bin_label, (min_val, max_val) in sorted_ranges:
            # 使用百分比字典获取正确的百分比
            percent = percentages_dict.get(bin_label, 0.0)
            print(f"  {bin_label}: 值域 [{min_val:.2f}, {max_val:.2f}], 占比 {percent:.2f}%")


# 主程序
if __name__ == "__main__":
    # 1. 从Excel文件中加载数据
    df = pd.read_excel('Cluster 21.xlsx')
    if len(df.columns) < 12:
        print(f"警告: Excel文件只有{len(df.columns)}列，使用所有可用列")
        num_cols = min(12, len(df.columns))
    else:
        num_cols = 12

    # 选择前N列作为变量
    data = df.iloc[:, :num_cols].copy()

    # 重命名列名为Var1到VarN
    data.columns = [f'Var{i + 1}' for i in range(num_cols)]

    print(f"成功从Excel文件加载数据，包含{len(data)}行和{len(data.columns)}列变量")
    print("数据摘要:")
    print(data.describe())

    # 2. 创建遗传算法优化器
    optimizer = GeneticDiscretizationOptimizer(
        data,
        min_bins=3,
        max_bins=10,
        min_support_thresh=0.1,
        rule_conf_thresh=0.5,
        population_size=30,
        generations=200,
        crossover_prob=0.7,
        mutation_prob=0.3
    )

    # 3. 运行优化
    best_config, best_fitness = optimizer.run_optimization()

    # 4. 可视化优化过程
    optimizer.visualize_optimization()

    # 5. 应用最佳配置
    discretized_data, bin_info = optimizer.apply_best_config()

    # 6. 可视化分箱结果
    visualize_binning(data, discretized_data, bin_info)

    # 7. 挖掘关联规则（使用优化器内部的方法确保一致性）
    transactions = optimizer.prepare_transactions(discretized_data)
    rules = optimizer.mine_association_rules(transactions)

    if not rules.empty:
        # 查找支持度最高的规则（适应度函数优化的目标）
        max_support_rule = rules.nlargest(1, 'support')
        if not max_support_rule.empty:
            max_support = max_support_rule['support'].values[0]
            print(f"\n最高支持度规则的支持度: {max_support:.4f} (与最佳适应度 {best_fitness:.4f} 一致)")

            # 打印最高支持度规则
            print("\n最高支持度规则:")
            print(max_support_rule)

            # 打印支持度大于50%的规则
            high_support_rules = rules[rules['support'] > 0.5]
            if not high_support_rules.empty:
                print("\n所有支持度大于50%的规则:")
                # 设置pandas显示选项以完整显示所有内容和所有列
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_colwidth', None)
                pd.set_option('display.width', None)
                print(high_support_rules)
                # 恢复默认显示设置
                pd.reset_option('display.max_columns')
                pd.reset_option('display.max_rows')
                pd.reset_option('display.max_colwidth')
                pd.reset_option('display.width')
            else:
                print("未找到支持度大于50%的规则")
        else:
            print("未找到关联规则")

    # 可视化规则质量
    if not rules.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('关联规则质量 (颜色表示提升度)')
        plt.grid(True)
        plt.savefig('association_rules_quality.png', dpi=300)
        plt.show()

    # 8. 保存结果
    try:
        if discretized_data is not None and not discretized_data.empty:
            combined_data = pd.concat([data, discretized_data.add_prefix('Disc_')], axis=1)
            combined_data.to_excel('optimized_discretized_data.xlsx', index=False)

        config_df = pd.DataFrame(best_config.items(), columns=['Variable', 'Bins'])
        config_df.to_excel('optimized_bins_config.xlsx', index=False)

        if rules is not None and not rules.empty:
            rules.to_excel('association_rules.xlsx', index=False)

        print("\n优化结果已保存")

    except Exception as e:
        print(f"保存结果时出错: {e}")

    print("\n优化完成!")