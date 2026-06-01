# 基于关联规则的参数重要性系统分析
import matplotlib
import matplotlib.pyplot as plt

# 解决中文显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体，然后使用DejaVu Sans
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import pandas as pd
import numpy as np
import networkx as nx
import re
import ast
from matplotlib import colormaps
import warnings


def extract_items_from_frozenset(fset_str):
    """从frozenset字符串中提取项"""
    try:
        # 提取{}中的内容
        items_str = re.search(r'{([^}]+)}', fset_str).group(1)

        # 去除额外的引号并分割项
        items = [item.strip().strip("'") for item in items_str.split(',')]
        return set(items)
    except Exception as e:
        print(f"无法解析frozenset: {fset_str}, 错误: {e}")
        return set()


def analyze_variable_importance(rules_df, bin_info):
    """在关联规则框架下系统分析变量重要性"""
    # 1. 准备变量列表
    all_variables = list(bin_info.keys())

    # 调试信息
    print(f"总变量数: {len(all_variables)}")
    print(f"规则总数: {len(rules_df)}")

    # 2. 初始化变量重要性数据字典
    importance_data = {
        'Variable': all_variables,
        'Rule_Participation': np.zeros(len(all_variables)),
        'Rule_Strength': np.zeros(len(all_variables)),
        'Consequent_Score': np.zeros(len(all_variables)),
        'Influence_Scope': np.zeros(len(all_variables)),
        'Network_Centrality': np.zeros(len(all_variables)),
        'Max_Support': np.zeros(len(all_variables)),
        'Dominant_Bin_Coverage': np.zeros(len(all_variables)),
        'Is_Unique': [False] * len(all_variables)  # 添加 Is_Unique 字段并初始化为 False
    }

    # 计数器用于调试
    successful_rules = 0
    failed_rules = 0

    # 2-1标记唯一值变量
    unique_vars = []  # 存储唯一值变量列表
    for i, var_name in enumerate(all_variables):
        if var_name in bin_info and 'bin_percentages' in bin_info[var_name]:
            bin_percentages = bin_info[var_name]['bin_percentages']
            if bin_percentages:
                # 如果只有一个分箱且覆盖率为100%，则为唯一值变量
                if len(bin_percentages) == 1 and list(bin_percentages.values())[0] >= 99.9:
                    importance_data['Is_Unique'][i] = True
                    unique_vars.append(var_name)
                    print(f"检测到唯一值变量: {var_name}, 覆盖率为: {list(bin_percentages.values())[0]:.2f}%")

    # 3. 规则参与度分析
    # 计算每个变量在前因和后果中出现的频率
    for _, rule in rules_df.iterrows():
        # 合并前因和后果中的所有项
        try:
            # 处理前因项
            if isinstance(rule['antecedents'], str) and rule['antecedents'].startswith('frozenset'):
                antecedents = extract_items_from_frozenset(rule['antecedents'])
            else:
                antecedents = rule['antecedents']

            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(antecedents, set):
                antecedents = set(antecedents)
            if not isinstance(consequents, set):
                consequents = set(consequents)

            all_items = antecedents.union(consequents)
            successful_rules += 1
        except Exception as e:
            # 如果解析失败，跳过该规则
            failed_rules += 1
            print(f"规则解析失败: {e}")
            print(f"规则内容: {rule}")
            continue

        # 从项中提取变量名
        for item in all_items:
            # 改进的变量名提取方法
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)

            # 如果变量在重要变量列表中
            if var_name in all_variables:
                idx = all_variables.index(var_name)
                importance_data['Rule_Participation'][idx] += 1

    # 打印调试信息
    print(f"成功解析的规则数: {successful_rules}/{len(rules_df)}")
    print(f"解析失败的规则数: {failed_rules}")

    # 4. 规则强度分析
    # 计算每个变量参与规则的平均支持度和置信度
    variable_strength = {}
    for _, rule in rules_df.iterrows():
        try:
            # 处理前因项
            if isinstance(rule['antecedents'], str) and rule['antecedents'].startswith('frozenset'):
                antecedents = extract_items_from_frozenset(rule['antecedents'])
            else:
                antecedents = rule['antecedents']

            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(antecedents, set):
                antecedents = set(antecedents)
            if not isinstance(consequents, set):
                consequents = set(consequents)

            all_items = antecedents.union(consequents)
        except:
            continue

        for item in all_items:
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)

            if var_name not in variable_strength:
                variable_strength[var_name] = {
                    'support_sum': 0,
                    'confidence_sum': 0,
                    'count': 0
                }

            try:
                # 确保规则中有support和confidence字段
                variable_strength[var_name]['support_sum'] += rule['support']
                variable_strength[var_name]['confidence_sum'] += rule['confidence']
                variable_strength[var_name]['count'] += 1
            except KeyError as ke:
                print(f"规则中缺失字段: {ke}")
                print(f"规则字段: {rule.keys()}")
                continue

    # 打印规则强度统计
    strength_count = 0
    for var_name, data in variable_strength.items():
        if var_name in all_variables:
            idx = all_variables.index(var_name)
            if data['count'] > 0:
                avg_support = data['support_sum'] / data['count']
                avg_confidence = data['confidence_sum'] / data['count']
                # 组合支持度和置信度作为规则强度
                importance_data['Rule_Strength'][idx] = avg_support * avg_confidence
                strength_count += 1

    print(f"成功计算规则强度的变量数: {strength_count}/{len(all_variables)}")

    # 5. 后果影响力分析
    consequent_count = 0
    for _, rule in rules_df.iterrows():
        try:
            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(consequents, set):
                consequents = set(consequents)
        except:
            continue

        for item in consequents:
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)

            if var_name in all_variables:
                idx = all_variables.index(var_name)
                importance_data['Consequent_Score'][idx] += 1
                consequent_count += 1

    print(f"发现变量作为后果的次数: {consequent_count}")

    # 6. 影响范围分析
    # 计算每个变量影响的其他变量数量
    # 创建变量共现字典
    variable_cooccurrence = {var: set() for var in all_variables}

    # 遍历所有规则，记录变量共现
    for _, rule in rules_df.iterrows():
        antecedent_vars = set()
        consequent_vars = set()

        try:
            # 处理前因项
            if isinstance(rule['antecedents'], str) and rule['antecedents'].startswith('frozenset'):
                antecedents = extract_items_from_frozenset(rule['antecedents'])
            else:
                antecedents = rule['antecedents']

            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(antecedents, set):
                antecedents = set(antecedents)
            if not isinstance(consequents, set):
                consequents = set(consequents)
        except:
            continue

        for item in antecedents:
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)
            antecedent_vars.add(var_name)

        for item in consequents:
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)
            consequent_vars.add(var_name)

        # 在规则中出现的所有变量
        all_rule_vars = antecedent_vars.union(consequent_vars)

        # 更新每个变量的影响范围
        for var in all_rule_vars:
            if var in variable_cooccurrence:
                variable_cooccurrence[var] = variable_cooccurrence[var].union(all_rule_vars)

    # 计算影响范围大小（减去自身）
    for var_name in all_variables:
        idx = all_variables.index(var_name)
        # 影响范围大小（连接的其他变量数量）
        connected_vars = variable_cooccurrence.get(var_name, set())
        influence_scope = len(connected_vars) - (1 if var_name in connected_vars else 0)
        importance_data['Influence_Scope'][idx] = max(0, influence_scope)

    print(f"影响范围分析完成，平均影响范围: {importance_data['Influence_Scope'].mean():.2f}")

    # 7. 网络中心度分析
    # 创建变量网络图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(all_variables)

    # 添加边（变量共现）
    for _, rule in rules_df.iterrows():
        # 提取规则中的所有变量
        rule_vars = set()
        try:
            # 处理前因项
            if isinstance(rule['antecedents'], str) and rule['antecedents'].startswith('frozenset'):
                antecedents = extract_items_from_frozenset(rule['antecedents'])
            else:
                antecedents = rule['antecedents']

            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(antecedents, set):
                antecedents = set(antecedents)
            if not isinstance(consequents, set):
                consequents = set(consequents)
        except:
            continue

        for item in antecedents.union(consequents):
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)
            rule_vars.add(var_name)

        # 为所有变量对添加边（或增加权重）
        rule_vars = list(rule_vars)
        for i in range(len(rule_vars)):
            for j in range(i + 1, len(rule_vars)):
                if rule_vars[i] != rule_vars[j]:
                    if G.has_edge(rule_vars[i], rule_vars[j]):
                        G[rule_vars[i]][rule_vars[j]]['weight'] += rule['support']
                    else:
                        G.add_edge(rule_vars[i], rule_vars[j], weight=rule['support'])

    # 计算特征向量中心度
    centrality = {}
    try:
        if G.number_of_nodes() > 0:
            # 检查是否连通图
            if nx.is_connected(G):
                # 如果是连通图，使用特征向量中心度
                centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=1000)
                print(f"特征向量中心度计算完成，节点数: {len(centrality)}")
            else:
                print("网络为不连通图，使用替代算法...")
                # 使用加权度中心性（所有边权重之和）
                centrality = {}
                for node in G.nodes():
                    # 获取节点权重和
                    weight_sum = sum(G[node][neighbor]['weight'] for neighbor in G[node])
                    centrality[node] = weight_sum
                print(f"使用加权度中心性替代")
        else:
            print("网络中没有节点，跳过中心度计算")
    except Exception as e:
        print(f"中心度计算失败: {e}")
        # 使用简单节点度作为备用
        centrality = {node: G.degree(node) for node in G.nodes()}

    # 对中心度进行归一化
    if centrality:
        max_cent = max(centrality.values())
        min_cent = min(centrality.values())
        range_cent = max_cent - min_cent

        if range_cent > 0:
            centrality = {k: (v - min_cent) / range_cent for k, v in centrality.items()}
        else:
            # 所有值相同，设为0.5
            centrality = {k: 0.5 for k in centrality.keys()}

    # 分配中心度值到重要性数据
    for var_name in all_variables:
        idx = all_variables.index(var_name)
        importance_data['Network_Centrality'][idx] = centrality.get(var_name, 0)

    # 8. 规则支持度分析
    # 找出每个变量参与的最高支持度规则
    max_support_per_var = {var: 0 for var in all_variables}

    for _, rule in rules_df.iterrows():
        # 提取规则中的所有变量
        rule_vars = set()
        try:
            # 处理前因项
            if isinstance(rule['antecedents'], str) and rule['antecedents'].startswith('frozenset'):
                antecedents = extract_items_from_frozenset(rule['antecedents'])
            else:
                antecedents = rule['antecedents']

            # 处理后果项
            if isinstance(rule['consequents'], str) and rule['consequents'].startswith('frozenset'):
                consequents = extract_items_from_frozenset(rule['consequents'])
            else:
                consequents = rule['consequents']

            # 确保都是集合类型
            if not isinstance(antecedents, set):
                antecedents = set(antecedents)
            if not isinstance(consequents, set):
                consequents = set(consequents)
        except:
            continue

        for item in antecedents.union(consequents):
            parts = str(item).split('_')
            if len(parts) >= 2:
                var_name = parts[0]
            else:
                var_name = str(item)
            rule_vars.add(var_name)

        # 更新每个变量的最大支持度
        for var in rule_vars:
            if var in max_support_per_var:
                if rule['support'] > max_support_per_var[var]:
                    max_support_per_var[var] = rule['support']

    for var_name, support in max_support_per_var.items():
        idx = all_variables.index(var_name)
        importance_data['Max_Support'][idx] = support

    print(f"最大支持度分析完成，平均最大支持度: {importance_data['Max_Support'].mean():.4f}")

    # 9. 主分箱覆盖率分析
    # 9-1. 对唯一值变量应用特殊处理 - 确保所有指标达到最高值
    if unique_vars:
        print("\n应用唯一值变量特殊处理...")
        unique_indices = [i for i, var in enumerate(importance_data['Variable']) if var in unique_vars]

        # 获取各项指标的全局最大值
        global_max = {
            'Rule_Participation': max(importance_data['Rule_Participation']),
            'Rule_Strength': max(importance_data['Rule_Strength']),
            'Consequent_Score': max(importance_data['Consequent_Score']),
            'Influence_Scope': max(importance_data['Influence_Scope']),
            'Network_Centrality': max(importance_data['Network_Centrality']),
            'Max_Support': max(importance_data['Max_Support']),
            'Dominant_Bin_Coverage': max(importance_data['Dominant_Bin_Coverage'])
        }

        # 为唯一值变量设置全局最大值（而非固定1.0）
        importance_data['Rule_Participation'][unique_indices] = global_max['Rule_Participation']
        importance_data['Rule_Strength'][unique_indices] = global_max['Rule_Strength']
        importance_data['Consequent_Score'][unique_indices] = global_max['Consequent_Score']
        importance_data['Influence_Scope'][unique_indices] = global_max['Influence_Scope']
        importance_data['Network_Centrality'][unique_indices] = global_max['Network_Centrality']
        importance_data['Max_Support'][unique_indices] = global_max['Max_Support']
        importance_data['Dominant_Bin_Coverage'][unique_indices] = global_max['Dominant_Bin_Coverage']

        print(f"已为{len(unique_vars)}个唯一值变量设置指标值为各指标全局最大值")

        # 现在计算主分箱覆盖率（确保唯一值处理不会被覆盖）
    for var_name in all_variables:
        idx = all_variables.index(var_name)
        # 获取该变量的分箱信息
        if var_name in bin_info and 'bin_percentages' in bin_info[var_name]:
            bin_percentages = bin_info[var_name]['bin_percentages']
            if bin_percentages:
                # 找出覆盖率最高的分箱
                max_coverage = max(bin_percentages.values())
                # 只更新唯一值变量尚未设置的值
                if not (var_name in unique_vars and importance_data['Dominant_Bin_Coverage'][idx] == 100.0):
                    importance_data['Dominant_Bin_Coverage'][idx] = max_coverage
            else:
                if not (var_name in unique_vars and importance_data['Dominant_Bin_Coverage'][idx] == 100.0):
                    importance_data['Dominant_Bin_Coverage'][idx] = 0
        else:
            if not (var_name in unique_vars and importance_data['Dominant_Bin_Coverage'][idx] == 100.0):
                importance_data['Dominant_Bin_Coverage'][idx] = 0

    print(
        f"主分箱覆盖率分析完成，平均覆盖率: {np.mean([x for x in importance_data['Dominant_Bin_Coverage'] if x > 0]):.2f}%")

    # 10. 创建重要性DataFrame
    importance_df = pd.DataFrame(importance_data)

    # 11. 计算综合重要性得分（加权平均）
    # 为各项指标分配权重（可根据分析需求调整）
    weights = {
        'Rule_Participation': 0.15,
        'Rule_Strength': 0.20,
        'Consequent_Score': 0.20,
        'Influence_Scope': 0.10,
        'Network_Centrality': 0.15,
        'Max_Support': 0.10,
        'Dominant_Bin_Coverage': 0.10
    }

    # 对指标进行归一化
    normalized_metrics = {}
    for metric in weights.keys():
        max_val = importance_df[metric].max()
        min_val = importance_df[metric].min()

        # 处理所有值相同的情况
        if max_val == min_val:
            # 所有值相同，全部设为1.0（最高分）
            normalized_metrics[metric] = np.ones_like(importance_df[metric])
        else:
            normalized_metrics[metric] = (importance_df[metric] - min_val) / (max_val - min_val)

    # 计算综合重要性得分
    importance_df['Overall_Importance'] = 0
    for metric, weight in weights.items():
        importance_df['Overall_Importance'] += normalized_metrics[metric] * weight

    # 按综合重要性排序
    importance_df = importance_df.sort_values('Overall_Importance', ascending=False)

    # 打印指标统计摘要
    print("\n指标统计摘要:")
    for col in importance_df.columns:
        if col != 'Variable' and col != 'Overall_Importance':
            mean_val = importance_df[col].mean()
            max_val = importance_df[col].max()
            min_val = importance_df[col].min()
            print(f"{col}: 均值={mean_val:.4f}, 最大值={max_val:.4f}, 最小值={min_val:.4f}")

    # 单独打印综合重要性
    mean_importance = importance_df['Overall_Importance'].mean()
    max_importance = importance_df['Overall_Importance'].max()
    min_importance = importance_df['Overall_Importance'].min()
    print(f"Overall_Importance: 均值={mean_importance:.4f}, 最大值={max_importance:.4f}, 最小值={min_importance:.4f}")

    # 打印唯一值变量的特殊处理信息
    unique_mask = importance_df['Is_Unique'] == True
    if unique_mask.any():
        # 修复索引问题
        unique_df = importance_df.loc[unique_mask, ['Variable', 'Overall_Importance']]
        print("\n唯一值变量重要性总结:")
        print(unique_df)

    return importance_df, G


def visualize_variable_importance(importance_df):
    """
    可视化变量重要性结果

    参数:
    importance_df -- 包含变量重要性指标的DataFrame
    """
    # 排序变量（按重要性升序，以便条形图从下到上显示）
    importance_df = importance_df.sort_values('Overall_Importance', ascending=True)

    # 识别唯一值变量
    if 'Is_Unique' in importance_df.columns:
        unique_mask = importance_df['Is_Unique'] == True
        unique_vars = set(importance_df.loc[unique_mask, 'Variable'])
    else:
        unique_vars = set()

    # 创建颜色列表 - 唯一值变量用红色突出显示
    colors = ['steelblue' if var in unique_vars else 'steelblue' for var in importance_df['Variable']]

    # 转换为百分比显示
    importance_df['Importance_Percent'] = importance_df['Overall_Importance'] * 100

    # 绘制综合重要性
    plt.figure(figsize=(14, 10))
    bars = plt.barh(importance_df['Variable'], importance_df['Importance_Percent'],
                    color=colors, edgecolor='black', alpha=0.7)

    # 添加每个条形的值
    for i, (bar, percent) in enumerate(zip(bars, importance_df['Importance_Percent'])):
        # 显示条形值
        plt.text(percent + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{percent:.2f}%',
                 ha='left', va='center', fontsize=16)

        # 如果是唯一值变量，添加星标
        if importance_df.iloc[i]['Variable'] in unique_vars:
            plt.scatter(percent * 0.5, bar.get_y() + bar.get_height() / 2,
                        s=200, marker='*', color='grey', zorder=5)

    # 为唯一值变量添加图例（金色星标）
    if unique_vars:
        # 使用金色(grey)星号标记，无连接线
        plt.plot([], [], color='grey', marker='*',
                 markersize=10, linestyle='None',
                 label='Unique-valued variable')
        plt.legend(loc='lower right', fontsize=16)

    plt.xlabel('Composite Importance Score (%)', fontsize=18)
    plt.ylabel('Variable', fontsize=18)
    plt.title('Composite Importance Ranking of Variables', fontsize=18, pad=20)
    # 设置坐标轴刻度标签字体大小
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.xlim(0, 110)  # 为标注留出空间
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('variable_importance_overall.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制各项指标雷达图
    metrics = ['Rule_Participation', 'Rule_Strength', 'Consequent_Score',
               'Influence_Scope', 'Network_Centrality', 'Max_Support',
               'Dominant_Bin_Coverage']

    # 选择Top 5变量进行雷达图可视化
    if len(importance_df) > 5:
        top_vars = importance_df.head(5)
    else:
        top_vars = importance_df

    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # 选择颜色
    colors = ['b', 'g', 'r', 'c', 'm']

    for idx, row in enumerate(top_vars.itertuples()):
        # 获取指标值并归一化
        values = [getattr(row, m) for m in metrics]
        max_val = max(values) if max(values) > 0 else 1
        normalized = [v / max_val for v in values]
        normalized += normalized[:1]  # 闭合多边形

        # 绘制雷达图
        ax.plot(angles, normalized, 'o-', color=colors[idx], linewidth=2,
                label=row.Variable)
        ax.fill(angles, normalized, color=colors[idx], alpha=0.1)

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_yticklabels([])
    plt.title('顶级变量的多维重要性分析', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.savefig('variable_importance_radar.png', dpi=300)
    plt.show()


def visualize_influence_network(G, importance_df, bin_info):
    """
    可视化变量影响网络

    参数:
    G -- 变量网络图
    importance_df -- 包含变量重要性指标的DataFrame
    bin_info -- 分箱信息字典
    """
    try:
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(18, 12), dpi=100)

        # 检查网络是否为空
        if len(G.nodes) == 0:
            print("无法绘制影响网络：没有有效节点")
            return

        # 创建节点大小基于综合重要性
        node_size = {}
        for var in importance_df['Variable']:
            score = importance_df.loc[importance_df['Variable'] == var, 'Overall_Importance'].values
            if len(score) > 0:
                node_size[var] = score[0] * 2000 + 100
            else:
                node_size[var] = 100

        # 设置默认节点大小
        for node in G.nodes:
            if node not in node_size:
                node_size[node] = 100

        # 创建节点颜色基于主分箱覆盖率
        bin_coverage = {}
        for var in G.nodes:
            if var in bin_info and 'bin_percentages' in bin_info[var]:
                bin_percentages = bin_info[var]['bin_percentages']
                if bin_percentages:
                    # 取主分箱的覆盖率（最大值）
                    max_coverage = max(bin_percentages.values())
                    bin_coverage[var] = max_coverage
                    continue
            bin_coverage[var] = 0

        # 计算最大覆盖值，避免除以零
        all_cov = list(bin_coverage.values())
        if all_cov:
            max_cov = max(all_cov)
            if max_cov <= 0:
                max_cov = 1
        else:
            max_cov = 1

        cmap = plt.get_cmap('viridis')
        node_colors = {}
        for node in G.nodes:
            cov = bin_coverage.get(node, 0)
            # 归一化覆盖率到0-1范围
            normalized_cov = cov / max_cov
            node_colors[node] = cmap(normalized_cov)

        # 绘制网络图
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # 使用弹簧布局

        # 创建网络图的Axes
        ax = fig.add_axes([0.1, 0.15, 0.7, 0.8])  # [left, bottom, width, height]

        # 绘制边
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=0.5, alpha=0.3)

        # 绘制节点
        sizes = [node_size[node] for node in G.nodes]
        colors = [node_colors[node] for node in G.nodes]

        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=colors, node_size=sizes
        )

        # 添加标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        # 创建颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
        sm.set_array([])

        # 创建颜色条专属的坐标轴
        cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

        # 添加颜色条
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('主分箱覆盖率 (%)', fontsize=14)

        ax.set_title('变量影响网络', fontsize=14)
        ax.axis('off')

        plt.savefig('variable_influence_network.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"在可视化影响网络时出错: {e}")
        import traceback
        traceback.print_exc()


def save_importance_report(importance_df):
    """保存变量重要性报告"""
    # 创建Excel编写器
    with pd.ExcelWriter('variable_importance_report.xlsx') as writer:
        # 保存原始数据
        importance_df.to_excel(writer, sheet_name='原始数据', index=False)

        # 创建带有说明的解释表
        explanation = pd.DataFrame({
            '指标': [
                'Rule_Participation',
                'Rule_Strength',
                'Consequent_Score',
                'Influence_Scope',
                'Network_Centrality',
                'Max_Support',
                'Dominant_Bin_Coverage',
                'Overall_Importance'
            ],
            '说明': [
                '变量在所有规则中出现的次数，表示其在规则系统中的活跃程度',
                '基于变量参与的规则的平均支持度和置信度计算的强度指标',
                '变量作为规则后果的频率，表示其对规则结果的贡献程度',
                '与该变量相关联的其他变量数量，表示其在系统中的影响范围',
                '基于变量在网络中的位置计算的中心度指标',
                '该变量参与规则的最高支持度，表示其最显著关联的代表性',
                '变量主分箱在数据中的覆盖率，表示其在数据中的主导程度',
                '综合各项指标计算的重要性得分（加权平均）'
            ]
        })
        explanation.to_excel(writer, sheet_name='指标解释', index=False)

        # 创建排序后的数据表（按综合重要性）
        sorted_df = importance_df.sort_values('Overall_Importance', ascending=False)
        sorted_df.to_excel(writer, sheet_name='综合排名', index=False)

        # 创建重要性分类表格
        def classify_importance(score):
            if score > 0.8:
                return 'Critical Importance'
            elif score > 0.6:
                return 'High Importance'
            elif score > 0.4:
                return 'Medium Importance'
            elif score > 0.2:
                return 'Normal Importance'
            else:
                return 'Low Importance'

        classified = sorted_df.copy()
        classified['Importance Level'] = classified['Overall_Importance'].apply(classify_importance)
        classified.to_excel(writer, sheet_name='重要性分类', index=False)

    print("变量重要性报告已保存至 variable_importance_report.xlsx")


if __name__ == "__main__":
    # 1. 加载关联规则数据
    try:
        rules_df = pd.read_excel('Cluster 24_association_rules.xlsx')
        print("关联规则文件加载成功!")
        print(f"包含 {len(rules_df)} 条规则")

        # 打印列名和前几条规则
        print(f"列名: {rules_df.columns.tolist()}")
        print("前2条规则:")
        print(rules_df.iloc[:2][['antecedents', 'consequents', 'support', 'confidence']])
    except Exception as e:
        print(f"加载关联规则文件出错: {e}")
        # 创建空DataFrame
        rules_df = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
        print("使用空规则数据框架继续进行分析...")

        # 2. 加载分箱信息（从优化后的数据中）
    bin_info = {}
    try:
        # 直接使用discretized_data计算分箱信息
        discretized_data = pd.read_excel('Cluster 24_optimized_discretized_data.xlsx')
        # 提取以'Disc_'开头的前12列
        disc_cols = [col for col in discretized_data.columns if col.startswith('Disc_')][:12]

        for col in disc_cols:
            # 获取原始变量名
            orig_col = col.replace('Disc_', '')

            # 获取分箱唯一值
            bins = discretized_data[col].unique()
            value_ranges = {}

            # 统计每个分箱的频率（百分比）
            bin_counts = discretized_data[col].value_counts(normalize=True) * 100
            bin_percentages = bin_counts.to_dict()

            # 创建值域字典
            for bin_val in bins:
                # 尝试提取数值范围
                if ':' in str(bin_val):
                    # 分箱标签中包含范围信息
                    range_str = bin_val.split(':')[1].strip(' []')
                    min_val, max_val = map(float, range_str.split(', '))
                    value_ranges[bin_val] = (min_val, max_val)
                else:
                    # 常量值或单个值
                    try:
                        num_val = float(bin_val.split('_')[-1])
                        value_ranges[bin_val] = (num_val, num_val)
                    except:
                        value_ranges[bin_val] = (0, 0)

            # 保存分箱信息
            bin_info[orig_col] = {
                'value_ranges': value_ranges,
                'bin_percentages': bin_percentages
            }

        print("分箱信息成功重建!")
        # 打印分箱信息示例
        print("\n分箱信息示例:")
        for var, info in list(bin_info.items())[:3]:
            print(f"变量: {var}")
            print(f"  值范围: {info['value_ranges']}")
            print(f"  分箱百分比: {info['bin_percentages']}")
    except Exception as e:
        print(f"加载分箱配置出错: {e}")
        # 创建模拟数据作为备选
        bin_info = {
            f"Var{i + 1}": {
                'value_ranges': {
                    f"Var{i + 1}_Bin{j}": (j * 10, j * 10 + 10) for j in range(3)
                },
                'bin_percentages': {
                    f"Var{i + 1}_Bin{j}": np.random.randint(20, 80) for j in range(3)
                }
            } for i in range(12)
        }
        print("使用模拟分箱信息继续进行分析...")

    # 3. 分析变量重要性
    importance_df, network_graph = analyze_variable_importance(rules_df, bin_info)

    # 4. 可视化结果
    visualize_variable_importance(importance_df)

    # 仅在网络图有节点时尝试可视化
    if network_graph is not None and network_graph.number_of_nodes() > 0:
        visualize_influence_network(network_graph, importance_df, bin_info)
    else:
        print("网络图中无有效节点，跳过影响网络可视化")

    # 5. 保存结果
    save_importance_report(importance_df)

    print("\n变量重要性分析完成!")