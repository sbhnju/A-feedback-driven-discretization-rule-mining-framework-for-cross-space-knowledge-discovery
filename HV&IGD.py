import numpy as np
import pandas as pd
from pathlib import Path

# 需要安装：
# pip install pandas openpyxl pymoo

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================
# 1. 文件路径：按你的实际文件名修改
# =========================
original_file = r"Original_ParetoGrid_Clustered.xlsx"
reference_file = r"Global_Pareto_Solutions.xlsx"

# 如果你已经有 guided set 的 Excel 文件，就填入路径；否则保持 None
guided_file = r"Guided_Set_Global_Pareto.xlsx"   # 例如：r"Guided_Set.xlsx"
# guided_file = None

# 如果 Excel 不是第一个 sheet，可以改成具体 sheet 名或 sheet 索引
sheet_name = 0


# =========================
# 2. 固定列名
# =========================
OBJ_COLS = ["EUI", "PPD", "UDI"]


# =========================
# 3. 读取 Excel
# =========================
def read_objectives(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    missing = [c for c in OBJ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} 缺少目标列: {missing}")

    obj = df[OBJ_COLS].copy()
    obj = obj.dropna().reset_index(drop=True)
    return obj


# =========================
# 4. 非支配排序：保留非支配前沿
# =========================
def get_non_dominated(points):
    """
    points: numpy array, shape = (n_points, n_obj), all objectives in minimization form
    """
    nd_idx = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[nd_idx]


# =========================
# 5. 统一归一化 + UDI 转最小化
# =========================
def normalize_to_minimization(df, bounds):
    """
    df: DataFrame with columns EUI, PPD, UDI
    bounds: {
        "EUI": (min, max),
        "PPD": (min, max),
        "UDI": (min, max)
    }

    返回：
        numpy array, shape=(n, 3)
        顺序为 [EUI_min, PPD_min, UDI_min]
    """
    eui_min, eui_max = bounds["EUI"]
    ppd_min, ppd_max = bounds["PPD"]
    udi_min, udi_max = bounds["UDI"]

    def safe_norm(x, xmin, xmax):
        if np.isclose(xmax, xmin):
            return np.zeros_like(x, dtype=float)
        return (x - xmin) / (xmax - xmin)

    eui_norm = safe_norm(df["EUI"].to_numpy(dtype=float), eui_min, eui_max)
    ppd_norm = safe_norm(df["PPD"].to_numpy(dtype=float), ppd_min, ppd_max)
    udi_norm = safe_norm(df["UDI"].to_numpy(dtype=float), udi_min, udi_max)

    # UDI 原本是 maximize，转成 minimization
    udi_min_form = 1.0 - udi_norm

    arr = np.column_stack([eui_norm, ppd_norm, udi_min_form])

    # 防止极小数值误差越界
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# =========================
# 6. 计算归一化边界
# =========================
def build_bounds(dataframes):
    """
    dataframes: list of DataFrame, 每个都含 EUI/PPD/UDI
    用所有比较集合 + reference front 共同确定统一归一化边界
    """
    merged = pd.concat(dataframes, axis=0, ignore_index=True)

    bounds = {
        "EUI": (merged["EUI"].min(), merged["EUI"].max()),
        "PPD": (merged["PPD"].min(), merged["PPD"].max()),
        "UDI": (merged["UDI"].min(), merged["UDI"].max()),
    }
    return bounds


# =========================
# 7. 计算单个集合的 HV / IGD
# =========================
def compute_metrics(candidate_points, reference_front, ref_point=np.array([1.1, 1.1, 1.1])):
    """
    candidate_points: numpy array, 已归一化、已是 minimization form
    reference_front: numpy array, 已归一化、已是 minimization form，且应为非支配前沿
    """
    # 候选集也保险起见再做一次非支配筛选
    candidate_nd = get_non_dominated(candidate_points)

    hv_indicator = HV(ref_point=ref_point)
    igd_indicator = IGD(reference_front)

    hv_value = hv_indicator(candidate_nd)
    igd_value = igd_indicator(candidate_nd)

    return hv_value, igd_value, len(candidate_nd)


# =========================
# 8. 主程序
# =========================
def main():
    # 读取数据
    original_df = read_objectives(original_file, sheet_name=sheet_name)
    reference_df = read_objectives(reference_file, sheet_name=sheet_name)

    guided_df = None
    if guided_file is not None and str(guided_file).strip() != "":
        guided_df = read_objectives(guided_file, sheet_name=sheet_name)

    # 用 reference front + 所有待比较集合，共同确定归一化边界
    dfs_for_bounds = [original_df, reference_df]
    if guided_df is not None:
        dfs_for_bounds.append(guided_df)

    bounds = build_bounds(dfs_for_bounds)

    # 归一化并转成 minimization
    original_norm = normalize_to_minimization(original_df, bounds)
    reference_norm = normalize_to_minimization(reference_df, bounds)

    if guided_df is not None:
        guided_norm = normalize_to_minimization(guided_df, bounds)
    else:
        guided_norm = None

    # reference front 再保险起见取一次非支配前沿
    reference_front = get_non_dominated(reference_norm)

    # 计算 Original Pareto set
    hv_original, igd_original, n_original = compute_metrics(
        candidate_points=original_norm,
        reference_front=reference_front,
        ref_point=np.array([1.1, 1.1, 1.1])
    )

    print("=== Original Pareto set ===")
    print(f"HV   = {hv_original:.10f}")
    print(f"IGD  = {igd_original:.10f}")
    print(f"NDS  = {n_original}")
    print()

    # 计算 Guided set（如果提供了 guided 文件）
    if guided_norm is not None:
        hv_guided, igd_guided, n_guided = compute_metrics(
            candidate_points=guided_norm,
            reference_front=reference_front,
            ref_point=np.array([1.1, 1.1, 1.1])
        )

        print("=== Guided set ===")
        print(f"HV   = {hv_guided:.10f}")
        print(f"IGD  = {igd_guided:.10f}")
        print(f"NDS  = {n_guided}")
        print()

        # 方便直接写进论文
        print("=== For manuscript/table ===")
        print(f"HV_original   = {hv_original:.10f}")
        print(f"IGD_original  = {igd_original:.10f}")
        print(f"N_original    = {n_original}")
        print(f"HV_guided     = {hv_guided:.10f}")
        print(f"IGD_guided    = {igd_guided:.10f}")
        print(f"N_guided      = {n_guided}")

    # 可选：保存结果
    results = [{
        "Set": "Original Pareto set",
        "HV": hv_original,
        "IGD": igd_original,
        "Non-dominated solutions": n_original
    }]

    if guided_norm is not None:
        results.append({
            "Set": "Guided set",
            "HV": hv_guided,
            "IGD": igd_guided,
            "Non-dominated solutions": n_guided
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel("HV_IGD_results.xlsx", index=False)
    print("\n结果已保存到: HV_IGD_results.xlsx")


if __name__ == "__main__":
    main()