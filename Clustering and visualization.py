import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl  # Import matplotlib for color customization
import os


def grid_based_clustering(input_file, output_file):
    """
    基于目标空间三维网格中心距离的聚类方法（3x3x3=27个区域）
    :param input_file: 输入Excel文件名
    :param output_file: 输出Excel文件名
    """
    try:
        # 第一步：使用pandas读取Excel文件
        df = pd.read_excel(input_file, header=0, engine='openpyxl')
        print(f"成功读取文件，总行数：{len(df)}")

        # 第二步：提取目标列（第13-15列，索引12-14）
        objectives = df.iloc[:, [12, 13, 14]].values

        # 第三步：计算目标空间分界点
        def get_grid_bins(values):
            min_val = np.min(values)
            max_val = np.max(values)
            return np.linspace(min_val, max_val, num=4)  # 3个区间需要4个分界点

        bins_obj0 = get_grid_bins(objectives[:, 0])  # 第13列（Min）
        bins_obj1 = get_grid_bins(objectives[:, 1])  # 第14列（Min）
        bins_obj2 = get_grid_bins(objectives[:, 2])  # 第15列（Max）

        # 第四步：计算网格中心点
        def get_grid_centers(bins):
            centers = []
            for i in range(len(bins) - 1):
                center = (bins[i] + bins[i + 1]) / 2
                centers.append(center)
            return np.array(centers)

        centers_obj0 = get_grid_centers(bins_obj0)  # 第13列网格中心
        centers_obj1 = get_grid_centers(bins_obj1)  # 第14列网格中心
        centers_obj2 = get_grid_centers(bins_obj2)  # 第15列网格中心（因为是最大化目标，需要反转）
        centers_obj2 = centers_obj2[::-1]  # 反转网格中心顺序，值越大中心索引越小

        # 第五步：创建27个网格中心点
        cluster_centers = np.zeros((27, 3))  # 27个中心点，每个3维
        cluster_idx = 0

        # 生成所有网格中心
        for i in range(3):  # 目标1的区间
            for j in range(3):  # 目标2的区间
                for k in range(3):  # 目标3的区间
                    # 注意目标3的中心顺序已经被反转
                    cluster_centers[cluster_idx] = [
                        centers_obj0[i],
                        centers_obj1[j],
                        centers_obj2[k]
                    ]
                    cluster_idx += 1

        # 第六步：计算每个解到所有网格中心的欧氏距离，分配最接近的网格
        labels = np.zeros(len(objectives), dtype=int)

        for idx, sol in enumerate(objectives):
            # 计算解到所有27个中心的距离
            distances = np.zeros(27)

            # 提取解的各个目标值（注意目标3是最大化，需要特殊处理）
            # 对于最小化目标，值越小越好；最大化目标，值越大越好
            # 但欧氏距离计算使用原始值
            obj1, obj2, obj3 = sol

            # 计算距离
            for center_idx in range(27):
                cx, cy, cz = cluster_centers[center_idx]
                # 计算欧氏距离
                distance = np.sqrt(
                    (obj1 - cx) ** 2 +
                                       (obj2 - cy) ** 2 +
                                                          (obj3 - cz) ** 2
                )
                distances[center_idx] = distance

            # 找到距离最小的中心
            labels[idx] = np.argmin(distances)

        # 第七步：保存带标签的结果到Excel
        result_df = df.copy()
        result_df[32] = labels.astype(int)  # 在最后一列添加聚类标签

        # 使用openpyxl引擎保存
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, header=False)

        # 第八步：增强可视化（增加另一个视角的二维散点图并改进颜色）
        plt.figure(figsize=(36, 14))
        plt.rcParams.update({
            'font.size': 16,  # 增加基础字体大小
            'axes.titlesize': 18,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 16,  # 增加x轴刻度字体
            'ytick.labelsize': 16  # 增加y轴刻度字体
        })

        # ========== 改进颜色映射：使用高区分度的离散颜色 ==========
        # 创建自定义颜色映射（27种更易区分的颜色）
        cmap = mpl.colors.ListedColormap([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
            '#5254a3', '#bd9e39'
        ])
        bounds = np.arange(-0.5, 27.5, 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # ------------- 二维散点图1: Objective 13 vs Objective 14 -------------
        ax1 = plt.subplot(1, 3, 1)
        scatter1 = ax1.scatter(objectives[:, 0], objectives[:, 1],
                               c=labels, cmap=cmap, norm=norm, alpha=0.9,
                               edgecolor='w', s=60)  # s=60增大点大小

        # 绘制网格线
        for b in bins_obj0:
            plt.axvline(b, color='gray', linestyle='--', linewidth=0.8)
        for b in bins_obj1:
            plt.axhline(b, color='gray', linestyle='--', linewidth=0.8)

        # 标记网格中心
        for i, center in enumerate(cluster_centers):
            plt.scatter(center[0], center[1], s=120, marker='*',
                        color='black', edgecolor='gold', linewidth=1.5)
        # 增加坐标轴数字大小
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_title('A: EUI vs PPD', fontsize=20, pad=15)
        ax1.set_xlabel('EUI (Minimize)', fontsize=18, labelpad=12)
        ax1.set_ylabel('PPD (Minimize)', fontsize=18, labelpad=12)

        cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=range(0, 27, 3), pad=0.02, aspect=30, fraction=0.18)
        cbar1.set_label('Cluster ID', fontsize=16, weight='bold')
        cbar1.ax.tick_params(labelsize=14)  # 设置颜色条刻度字体

        # ------------- 新增二维散点图2: Objective 13 vs Objective 15 -------------
        ax2 = plt.subplot(1, 3, 2)
        scatter2 = ax2.scatter(objectives[:, 0], objectives[:, 2],
                               c=labels, cmap=cmap, norm=norm, alpha=0.9,
                               edgecolor='w', s=60)  # s=60增大点大小

        # 绘制网格线
        for b in bins_obj0:
            plt.axvline(b, color='gray', linestyle='--', linewidth=0.8)
        for b in bins_obj2:  # 使用原始分界点（未反转）
            plt.axhline(b, color='gray', linestyle='--', linewidth=0.8)

        # 标记网格中心 (注意Objective 15使用原始中心位置)
        for i, center in enumerate(cluster_centers):
            # 使用原始中心点位置（不反转）
            plt.scatter(center[0], cluster_centers[i][2], s=120, marker='*',
                        color='black', edgecolor='gold', linewidth=1.5)

        # 增加坐标轴数字大小
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.set_title('B: EUI vs UDI', fontsize=20, pad=15)
        ax2.set_xlabel('EUI (Minimize)', fontsize=18, labelpad=12)
        ax2.set_ylabel('UDI (Maximize)', fontsize=18, labelpad=12)

        cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=range(0, 27, 3), pad=0.02, aspect=30, fraction=0.18)
        cbar2.set_label('Cluster ID', fontsize=16, weight='bold')
        cbar2.ax.tick_params(labelsize=14)  # 设置颜色条刻度字体

        # ------------- 三维散点图 -------------
        ax3d = plt.subplot(1, 3, 3, projection='3d')

        # 绘制三维散点 (使用改进后的颜色映射)
        scatter_3d = ax3d.scatter(
            objectives[:, 0],  # X轴：Obj13
            objectives[:, 1],  # Y轴：Obj14
            objectives[:, 2],  # Z轴：Obj15
            c=labels,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
            edgecolor='w',
            s=40
        )

        # 绘制网格中心
        for i, center in enumerate(cluster_centers):
            ax3d.scatter(center[0], center[1], center[2],
                         s=150, marker='*', color='black',
                         edgecolor='gold', linewidth=1.5)

        # 增大三维图的坐标轴数字大小
        ax3d.tick_params(axis='x', labelsize=14)
        ax3d.tick_params(axis='y', labelsize=14)
        ax3d.tick_params(axis='z', labelsize=14)

        # 添加图标题和标签（加大标题字体，增加标签间距）
        ax3d.set_title('C: 3D View with Cluster Centers', fontsize=20, pad=15)
        ax3d.set_xlabel('EUI\n(Minimize)', fontsize=18, labelpad=15)
        ax3d.set_ylabel('PPD\n(Minimize)', fontsize=18, labelpad=15)
        ax3d.set_zlabel('UDI\n(Maximize)', fontsize=18, labelpad=15)

        # 添加3D图的颜色条（尺寸与2D图一致）
        cbar3 = plt.colorbar(scatter_3d, ax=ax3d, ticks=range(0, 27, 3),
                             shrink=0.6, pad=0.1, aspect=30, fraction=0.18)
        cbar3.set_label('Cluster ID', fontsize=16, weight='bold')
        cbar3.ax.tick_params(labelsize=14)  # 设置颜色条刻度字体

        # 设置三维图的视觉参数（优化视角）
        ax3d.view_init(elev=25, azim=135)
        ax3d.grid(True, alpha=0.3)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

        # 添加整体标题（居中置于上方）
        plt.suptitle('Pareto Front Clustering: 3×3×3 Grid-Based Partition',
                     fontsize=24, weight='bold', y=0.98)

        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.25)
        plot_file = output_file.replace('.xlsx', '_side_by_side_visualization.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 第九步：输出每个聚类的边界信息
        # 创建边界信息字典
        cluster_boundaries = []

        # 重新生成网格索引
        cluster_idx = 0
        for i in range(3):  # 目标1的区间
            for j in range(3):  # 目标2的区间
                for k in range(3):  # 目标3的区间
                    # 目标3的边界需要反转
                    k_rev = 2 - k  # 反转索引

                    # 获取边界
                    obj0_min = bins_obj0[i]
                    obj0_max = bins_obj0[i + 1]
                    obj1_min = bins_obj1[j]
                    obj1_max = bins_obj1[j + 1]
                    obj2_min = bins_obj2[k_rev]
                    obj2_max = bins_obj2[k_rev + 1]

                    # 添加到边界信息
                    cluster_boundaries.append({
                        'Cluster ID': cluster_idx,
                        'EUI Min': obj0_min,
                        'EUI Max': obj0_max,
                        'PPD Min': obj1_min,
                        'PPD Max': obj1_max,
                        'UDI Min': obj2_min,
                        'UDI Max': obj2_max
                    })

                    cluster_idx += 1

        # 创建边界信息DataFrame
        boundaries_df = pd.DataFrame(cluster_boundaries)

        # 保存边界信息到Excel
        boundaries_file = output_file.replace('.xlsx', '_boundaries.xlsx')
        boundaries_df.to_excel(boundaries_file, index=False)

        # 保存边界信息到文本文件
        boundaries_txt_file = output_file.replace('.xlsx', '_boundaries.txt')
        with open(boundaries_txt_file, 'w') as f:
            f.write("Cluster Boundaries:\n")
            f.write("Format: Cluster ID | EUI Min - EUI Max | PPD Min - PPD Max | UDI Min - UDI Max\n")
            f.write("=" * 80 + "\n")

            for boundary in cluster_boundaries:
                f.write(f"Cluster {boundary['Cluster ID']}:\n")
                f.write(f"  EUI: {boundary['EUI Min']:.6f} to {boundary['EUI Max']:.6f}\n")
                f.write(f"  PPD: {boundary['PPD Min']:.6f} to {boundary['PPD Max']:.6f}\n")
                f.write(f"  UDI: {boundary['UDI Min']:.6f} to {boundary['UDI Max']:.6f}\n")
                f.write("-" * 60 + "\n")

        print(f"\n处理完成！共生成27个类别")
        print(f"带标签文件：{output_file}")
        print(f"可视化文件：{plot_file}")
        print(f"聚类边界文件：{boundaries_file} 和 {boundaries_txt_file}")
        print(f"网格中心位置（前5个）:")
        for i in range(5):
            print(f"Cluster {i}: {cluster_centers[i]}")
        print("注：目标3（最大化）的网格中心顺序已反转")

        # 打印边界信息摘要
        print("\n聚类边界摘要:")
        print("Cluster ID | EUI Min - EUI Max | PPD Min - PPD Max | UDI Min - UDI Max")
        print("=" * 80)
        for i in range(3):
            print(f"Cluster {i}:")
            print(f"  EUI: {cluster_boundaries[i]['EUI Min']:.6f} to {cluster_boundaries[i]['EUI Max']:.6f}")
            print(f"  PPD: {cluster_boundaries[i]['PPD Min']:.6f} to {cluster_boundaries[i]['PPD Max']:.6f}")
            print(f"  UDI: {cluster_boundaries[i]['UDI Min']:.6f} to {cluster_boundaries[i]['UDI Max']:.6f}")
            print("-" * 60)

    except Exception as e:
        print(f"\n处理失败：{str(e)}")
        import traceback
        traceback.print_exc()
        if isinstance(e, FileNotFoundError):
            print(f"请检查文件路径是否正确：{input_file}")
        elif isinstance(e, ValueError):
            print("可能的原因：列数不足32列或数据格式错误")
        print("建议：检查输入文件格式是否符合要求")


if __name__ == "__main__":
    input_file = "non_dominated_solutions_deduplicated.xlsx"  # 输入Excel文件
    output_file = "non_dominated_solutions_ParetoGrid_Clustered.xlsx"
    grid_based_clustering(input_file, output_file)