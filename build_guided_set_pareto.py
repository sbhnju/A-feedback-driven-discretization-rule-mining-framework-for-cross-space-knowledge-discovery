import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# 可选：如果安装了 pyarrow，就额外输出 parquet
try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


# =========================
# 1. 日志
# =========================
def setup_logging():
    logger = logging.getLogger("guided_pareto")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# =========================
# 2. Fenwick Tree（前缀最小值）
# =========================
class FenwickMin:
    def __init__(self, size: int):
        self.n = size
        self.tree = np.full(size + 1, np.inf, dtype=np.float64)

    def update(self, idx: int, value: float):
        while idx <= self.n:
            if value < self.tree[idx]:
                self.tree[idx] = value
            idx += idx & -idx

    def query(self, idx: int) -> float:
        res = np.inf
        while idx > 0:
            if self.tree[idx] < res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res


# =========================
# 3. 三目标精确非支配排序（最小化）
# =========================
def efficient_pareto_frontier_3d(points: np.ndarray, decimals: int | None = 10) -> np.ndarray:
    """
    对 shape=(n,3) 的目标矩阵做精确三目标非支配筛选
    所有目标都必须已经转成“越小越好”

    参数
    ----
    points : ndarray, shape=(n,3)
    decimals : 可选。用于轻微舍入，降低浮点误差影响。None 表示不舍入。

    返回
    ----
    mask : ndarray(bool), shape=(n,)
        True 表示该点为非支配点
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("efficient_pareto_frontier_3d 只支持 shape=(n, 3) 的三目标数据。")

    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 轻微舍入，减少极小浮点误差导致的“几乎相同点”问题
    work = np.round(points, decimals=decimals) if decimals is not None else points.copy()

    # 先按目标三元组去重
    unique_pts, inverse = np.unique(work, axis=0, return_inverse=True)

    # 按 f1, f2, f3 升序排序（都为最小化）
    order = np.lexsort((unique_pts[:, 2], unique_pts[:, 1], unique_pts[:, 0]))
    sorted_pts = unique_pts[order]

    # 对第二目标做坐标压缩
    _, rank0 = np.unique(sorted_pts[:, 1], return_inverse=True)
    ranks = rank0 + 1  # Fenwick tree 用 1-based 索引

    bit = FenwickMin(len(np.unique(sorted_pts[:, 1])))
    is_nd_sorted = np.ones(len(sorted_pts), dtype=bool)

    # 扫描
    for i in range(len(sorted_pts)):
        r = ranks[i]
        f3 = sorted_pts[i, 2]

        # 查询所有历史点中，f2 <= 当前点 的最小 f3
        best_f3 = bit.query(r)

        # 由于已按 f1 升序扫描，历史点天然满足 f1 <= 当前点
        # 若存在历史点使得 f2 <= 当前点 且 f3 <= 当前点，则当前点被支配
        if best_f3 <= f3:
            is_nd_sorted[i] = False

        # 更新当前点
        bit.update(r, f3)

    # 还原到 unique_pts 顺序
    is_nd_unique = np.zeros(len(unique_pts), dtype=bool)
    is_nd_unique[order] = is_nd_sorted

    # 再映射回原始点
    result = is_nd_unique[inverse]
    return result


# =========================
# 4. 读取 guided 文件
# =========================
def read_guided_files(input_files: list[Path], logger: logging.Logger) -> pd.DataFrame:
    required_cols = [
        "Var1", "Var2", "Var3", "Var4", "Var5", "Var6",
        "Var7", "Var8", "Var9", "Var10", "Var11", "Var12",
        "EUI", "PPD", "UDI", "Generation"
    ]

    dtype_map = {
        "Var1": "float64", "Var2": "float64", "Var3": "float64", "Var4": "float64",
        "Var5": "float64", "Var6": "float64", "Var7": "float64", "Var8": "float64",
        "Var9": "float64", "Var10": "float64", "Var11": "float64", "Var12": "float64",
        "EUI": "float64", "PPD": "float64", "UDI": "float64",
        "Generation": "Int64"
    }

    all_dfs = []

    logger.info("阶段1: 读取并合并所有输入文件...")

    for i, file_path in enumerate(input_files, start=1):
        if not file_path.exists():
            logger.warning(f"[{i}/{len(input_files)}] 文件不存在，跳过: {file_path.name}")
            continue

        logger.info(f"[{i}/{len(input_files)}] 读取文件: {file_path.name}")
        t0 = time.time()

        df = pd.read_excel(
            file_path,
            engine="openpyxl"
        )

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_path.name} 缺少列: {missing}")

        # 只保留需要的列
        df = df[required_cols].copy()

        # 类型转换
        for col, dt in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dt)
                except Exception:
                    # 保底处理
                    if col == "Generation":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    else:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

        # 删除目标值缺失行
        df = df.dropna(subset=["EUI", "PPD", "UDI"]).reset_index(drop=True)

        # 增加数据源列
        df["Data_Source"] = file_path.stem

        elapsed = time.time() - t0
        logger.info(f"文件处理完成: {len(df)} 行 | 用时 {elapsed:.2f} 秒")

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("没有成功读取任何 guided 文件。")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"所有文件合并完成: 总行数 = {len(combined):,}")

    mem_mb = combined.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"合并数据集内存使用量: {mem_mb:.2f} MB")

    return combined


# =========================
# 5. 计算 guided set 的全局非支配解
# =========================
def compute_guided_global_pareto(data: pd.DataFrame, logger: logging.Logger, decimals: int | None = 10) -> pd.DataFrame:
    if data.empty:
        raise ValueError("输入数据为空。")

    logger.info("阶段2: 计算 guided set 的全局帕累托前沿...")
    logger.info(f"开始全局帕累托计算 - 总样本数: {len(data):,}")

    t0 = time.time()

    # 提取目标值：EUI, PPD 最小化；UDI 最大化 -> 转成最小化
    points = data[["EUI", "PPD", "UDI"]].to_numpy(dtype=np.float64, copy=True)
    points[:, 2] = -points[:, 2]  # UDI 转最小化

    nd_mask = efficient_pareto_frontier_3d(points, decimals=decimals)

    result = data.copy()
    result["Is_Global_Pareto"] = nd_mask

    pareto_df = result[result["Is_Global_Pareto"]].copy().reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(
        f"全局帕累托计算完成: 解数量 = {len(pareto_df):,}/{len(result):,} "
        f"| 占比 = {len(pareto_df)/len(result):.2%} | 用时 = {elapsed:.2f} 秒"
    )

    return result, pareto_df


# =========================
# 6. 保存结果
# =========================
def save_results(all_data: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: Path, logger: logging.Logger):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出 guided set 全局非支配解
    pareto_excel = out_dir / "Guided_Set_Global_Pareto.xlsx"
    pareto_csv = out_dir / "Guided_Set_Global_Pareto.csv"

    logger.info("阶段3: 保存结果...")
    pareto_df.to_excel(pareto_excel, index=False, engine="openpyxl")
    pareto_df.to_csv(pareto_csv, index=False, encoding="utf-8-sig")

    logger.info(f"已保存 Excel: {pareto_excel}")
    logger.info(f"已保存 CSV  : {pareto_csv}")

    # 可选保存全部合并数据
    if HAS_PYARROW:
        all_parquet = out_dir / "Guided_Set_All_Merged.parquet"
        all_data.to_parquet(all_parquet, index=False)
        logger.info(f"已保存 Parquet: {all_parquet}")
    else:
        logger.info("未安装 pyarrow，跳过 parquet 输出。")

    # 输出一个摘要表
    summary = pd.DataFrame([
        {
            "Total_samples": len(all_data),
            "Global_pareto_samples": len(pareto_df),
            "Pareto_ratio": len(pareto_df) / len(all_data)
        }
    ])
    summary_file = out_dir / "Guided_Set_Summary.xlsx"
    summary.to_excel(summary_file, index=False, engine="openpyxl")
    logger.info(f"已保存摘要: {summary_file}")


# =========================
# 7. 主函数
# =========================
def main():
    logger = setup_logging()
    start_all = time.time()

    # 以脚本所在目录作为工作目录
    base_dir = Path(__file__).resolve().parent

    input_files = [
        base_dir / "Cluster 0_New.xlsx",
        base_dir / "Cluster 1_New.xlsx",
        base_dir / "Cluster 2_New.xlsx",
        base_dir / "Cluster 3_New.xlsx",
        base_dir / "Cluster 9_New.xlsx",
        base_dir / "Cluster 12_New.xlsx",
        base_dir / "Cluster 15_New.xlsx",
        base_dir / "Cluster 24_New.xlsx",
    ]

    output_dir = base_dir / "guided_set_output"

    logger.info("开始构建 guided set 的全局帕累托前沿...")

    combined_data = read_guided_files(input_files, logger=logger)
    all_data, global_pareto = compute_guided_global_pareto(
        combined_data,
        logger=logger,
        decimals=10  # 若不想舍入，可改成 None
    )
    save_results(all_data, global_pareto, out_dir=output_dir, logger=logger)

    elapsed = time.time() - start_all
    logger.info("=" * 60)
    logger.info("处理完成")
    logger.info(f"总样本数           : {len(all_data):,}")
    logger.info(f"全局非支配解数量   : {len(global_pareto):,}")
    logger.info(f"总用时             : {elapsed:.2f} 秒")
    logger.info(f"输出目录           : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()