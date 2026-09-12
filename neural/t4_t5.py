"""T4/T5 方向選擇性細胞識別

MaleCNS v1.0 包含完整的果蠅視覺系統,內建 T4/T5 方向選擇性細胞。
這些細胞是果蠅視覺系統裡最著名的方向選擇性細胞,Maisak et al. 2013 (Nature)
已用 GCaMP 電生理確立其方向調諧圖。

T4 亞型(共 ~6865 個):
  T4a/b/c/d 對應不同視覺野區域,各自有方向偏好

T5 亞型(共 ~6720 個):
  T5a/b/c/d 同樣分區,方向偏好與 T4 對應

合計 ~13,585 個方向選擇性細胞,是本實驗的核心測量對象。
"""

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path


def identify_t4_t5(ids):
    """從 MaleCNS annotations 識別所有 T4/T5 細胞

    Parameters
    ----------
    ids : np.ndarray of uint64
        已排序的神經元 bodyId(與 connectome 中的 node IDs 對應)

    Returns
    -------
    dict with keys:
        t4 : np.ndarray of int32  -- T4 細胞的 node indices
        t5 : np.ndarray of int32  -- T5 細胞的 node indices
        t4_t5 : np.ndarray of int32  -- 合併
        t4_subtypes : dict  -- 各 T4 亞型的 indices
        t5_subtypes : dict  -- 各 T5 亞型的 indices
        upstream : dict  -- 上stream 細胞(L1-L5, Mi1, Tm3 等)
        report : dict  -- 統計報告
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    annotations = feather.read_table(data_dir / "annotations.feather").to_pandas()
    annotations = annotations.set_index("bodyId")

    # 取得 ids 對應的 type(轉成 Python str list 避免 numpy object dtype 問題)
    types_raw = annotations.loc[ids, "type"].fillna("").astype(str).tolist()
    types = np.array(types_raw, dtype=object)
    n = len(ids)

    # 識別 T4 各亞型(用 list comprehension 避免 np.char.startswith 對 object dtype 的問題)
    t4_subtypes = {}
    for subtype in ["T4a", "T4b", "T4c", "T4d"]:
        mask = np.array([t.startswith(subtype) for t in types])
        t4_subtypes[subtype] = np.flatnonzero(mask).astype(np.int32)

    # 識別 T5 各亞型
    t5_subtypes = {}
    for subtype in ["T5a", "T5b", "T5c", "T5d"]:
        mask = np.array([t.startswith(subtype) for t in types])
        t5_subtypes[subtype] = np.flatnonzero(mask).astype(np.int32)

    # 合併所有 T4/T5
    t4 = np.concatenate([t4_subtypes[k] for k in ["T4a", "T4b", "T4c", "T4d"]])
    t5 = np.concatenate([t5_subtypes[k] for k in ["T5a", "T5b", "T5c", "T5d"]])
    t4_t5 = np.concatenate([t4, t5])

    # 也識別上游視覺細胞(供分析用)
    upstream = {}
    for cell_type in ["L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm9"]:
        # 精確比對(避免 Mi1 匹配到 Mi15)
        mask = np.array([t == cell_type for t in types])
        if mask.any():
            upstream[cell_type] = np.flatnonzero(mask).astype(np.int32)

    # 光感受器
    r1_6_mask = np.array([t.startswith("R1-R6") or t in ["R1", "R2", "R3", "R4", "R5", "R6"] for t in types])
    upstream["R1-R6"] = np.flatnonzero(r1_6_mask).astype(np.int32)

    r7_mask = np.array([t.startswith("R7") for t in types])
    upstream["R7"] = np.flatnonzero(r7_mask).astype(np.int32)

    r8_mask = np.array([t.startswith("R8") for t in types])
    upstream["R8"] = np.flatnonzero(r8_mask).astype(np.int32)

    report = {
        "release": "MaleCNS v1.0",
        "total_neurons": n,
        "t4_count": len(t4),
        "t5_count": len(t5),
        "t4_t5_total": len(t4_t5),
        "t4_subtypes": {k: len(v) for k, v in t4_subtypes.items()},
        "t5_subtypes": {k: len(v) for k, v in t5_subtypes.items()},
        "upstream_counts": {k: len(v) for k, v in upstream.items()},
        "biological_reference": "Maisak et al. 2013, Nature - T4/T5 direction selectivity",
        "expected_dsi_range": "0.6-0.8 (Maisak 2013, GCaMP electrophysiology)",
        "validated": False,
    }

    return {
        "t4": t4,
        "t5": t5,
        "t4_t5": t4_t5,
        "t4_subtypes": t4_subtypes,
        "t5_subtypes": t5_subtypes,
        "upstream": upstream,
        "report": report,
    }


if __name__ == "__main__":
    import json
    import sys
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DATA
    import numpy as np

    # 載入 neuron IDs
    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    print(f"Loaded {len(ids)} neuron IDs")

    result = identify_t4_t5(ids)
    print("\n" + "=" * 60)
    print("T4/T5 識別結果")
    print("=" * 60)
    print(json.dumps(result["report"], indent=2))

    print(f"\n--- T4 亞型分佈 ---")
    for k, v in result["t4_subtypes"].items():
        print(f"  {k}: {len(v)} cells")

    print(f"\n--- T5 亞型分佈 ---")
    for k, v in result["t5_subtypes"].items():
        print(f"  {k}: {len(v)} cells")

    print(f"\n--- 上游視覺細胞 ---")
    for k, v in result["upstream"].items():
        print(f"  {k}: {len(v)} cells")
