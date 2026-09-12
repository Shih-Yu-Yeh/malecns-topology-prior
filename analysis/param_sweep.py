"""參數掃描:找到讓 MaleCNS T4 DSI 接近 0.6-0.8 的參數組合

策略:
1. 固定 duration_ms=300(快速)
2. 掃描 mean_drive × temporal_freq(5×4=20 組合)
3. 用最佳組合再掃 spatial_freq(4 組合)
4. 最後用最佳參數跑完整 null model 對照(5 seeds)

公平性原則:所有參數對 MaleCNS / Config / ER 一視同仁
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neural.common import DATA, GRAPH
from neural.t4_t5 import identify_t4_t5
from neural.brain import MemoryBrain
import pyarrow.feather as feather
import pandas as pd


def get_mi1_spatial_layout(ids, t4_t5_result):
    ann = feather.read_table(DATA / "annotations.feather").to_pandas().set_index("bodyId")
    mi1 = t4_t5_result["upstream"]["Mi1"]
    mi1_indices, mi1_hex = [], []
    for idx in mi1:
        body_id = ids[idx]
        h1 = ann.loc[body_id, "assignedOlHex1"]
        h2 = ann.loc[body_id, "assignedOlHex2"]
        if pd.notna(h1) and pd.notna(h2):
            mi1_indices.append(idx)
            mi1_hex.append((float(h1), float(h2)))
    mi1_indices = np.array(mi1_indices, dtype=np.int32)
    mi1_hex = np.array(mi1_hex)
    mi1_xy = np.column_stack([
        mi1_hex[:, 0] - 0.5 * mi1_hex[:, 1],
        np.sqrt(3) / 2 * mi1_hex[:, 1]
    ])
    return mi1_indices, mi1_xy


def moving_wave_stimulus(mi1_xy, direction_deg, t_ms,
                          spatial_freq=0.05, temporal_freq=2.0,
                          mean_drive=12.0):
    direction_rad = np.deg2rad(direction_deg)
    dx = np.cos(direction_rad)
    dy = np.sin(direction_rad)
    projection = mi1_xy[:, 0] * dx + mi1_xy[:, 1] * dy
    phase = 2 * np.pi * temporal_freq * (t_ms / 1000.0)
    wave = mean_drive + mean_drive * np.sin(
        2 * np.pi * spatial_freq * projection - phase
    )
    return np.clip(wave, 0, None).astype(np.float32)


def run_single(brain, mi1_indices, mi1_xy, t4_t5_result, directions,
                duration_ms=300, mean_drive=12.0,
                spatial_freq=0.05, temporal_freq=2.0):
    """跑 8 方向,返回 T4 DSI 統計"""
    t4 = t4_t5_result["t4"]
    tonic = np.zeros(brain.n, dtype=np.float32)
    tonic[t4_t5_result["upstream"]["Mi4"]] = 3.0
    brain.tonic = tonic

    dummy_lum = np.zeros(len(brain.retina), dtype=np.float32)
    n_batches = int(duration_ms / 10)
    t4_rates_all = []

    for direction in directions:
        brain.reset(keep_memory=False)
        t4_counts = np.zeros(len(t4), dtype=np.int32)
        for b in range(n_batches):
            t_ms = b * 10
            mi1_drive = moving_wave_stimulus(
                mi1_xy, direction, t_ms,
                spatial_freq=spatial_freq,
                temporal_freq=temporal_freq,
                mean_drive=mean_drive
            )
            counts, _ = brain.step(
                dummy_lum, 10, learning=False,
                stimulation=[(mi1_indices, mi1_drive)]
            )
            t4_counts += counts[t4]
        t4_rate = t4_counts / (duration_ms / 1000)
        t4_rates_all.append(t4_rate)

    t4_rates_all = np.array(t4_rates_all)

    # DSI
    n_cells = t4_rates_all.shape[1]
    dsi = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        rates = t4_rates_all[:, i]
        if rates.sum() == 0:
            continue
        pref_idx = np.argmax(rates)
        null_idx = (pref_idx + 4) % 8
        r_pref = rates[pref_idx]
        r_null = rates[null_idx]
        denom = r_pref + r_null
        if denom > 0:
            dsi[i] = abs(r_pref - r_null) / denom

    active = t4_rates_all.sum(axis=0) > 0
    n_active = active.sum()
    dsi_active = dsi[active] if n_active > 0 else np.array([])

    return {
        "n_active": int(n_active),
        "dsi_median_active": float(np.median(dsi_active)) if len(dsi_active) > 0 else 0,
        "dsi_gt_06_pct": float((dsi_active > 0.6).mean() * 100) if len(dsi_active) > 0 else 0,
        "dsi_mean_active": float(dsi_active.mean()) if len(dsi_active) > 0 else 0,
        "t4_avg_rate": float(t4_rates_all.mean()),
    }


def main():
    print("=" * 70)
    print("參數掃描:找到讓 MaleCNS T4 DSI 接近 0.6-0.8 的參數")
    print("=" * 70)

    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    mi1_indices, mi1_xy = get_mi1_spatial_layout(ids, t4_t5_result)

    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    brain = MemoryBrain(GRAPH, eta=0.0)

    # Phase 1: 掃描 mean_drive × temporal_freq
    print("\n--- Phase 1: mean_drive × temporal_freq ---")
    drives = [8, 10, 12, 15, 20]
    freqs = [1, 2, 4, 8]
    results_phase1 = []

    print(f"\n{'drive':>6s} {'freq':>5s} {'active':>7s} {'DSI_med':>8s} {'DSI>0.6':>8s} {'avg_rate':>9s}")
    print("-" * 50)

    for drive in drives:
        for freq in freqs:
            result = run_single(
                brain, mi1_indices, mi1_xy, t4_t5_result, directions,
                duration_ms=300, mean_drive=drive,
                spatial_freq=0.05, temporal_freq=freq
            )
            results_phase1.append({
                "drive": drive, "freq": freq, **result
            })
            print(f"{drive:6d} {freq:5d} {result['n_active']:7d} "
                  f"{result['dsi_median_active']:8.3f} {result['dsi_gt_06_pct']:7.1f}% "
                  f"{result['t4_avg_rate']:9.2f}")

    # 找最佳組合(DSI 中位數最高 + active > 10)
    valid = [r for r in results_phase1 if r["n_active"] >= 10]
    best_phase1 = max(valid, key=lambda r: r["dsi_median_active"])
    print(f"\n✅ Phase 1 最佳: drive={best_phase1['drive']}, freq={best_phase1['freq']}")
    print(f"   DSI median = {best_phase1['dsi_median_active']:.3f}, "
          f"active = {best_phase1['n_active']}, DSI>0.6 = {best_phase1['dsi_gt_06_pct']:.1f}%")

    # Phase 2: 用最佳 drive×freq 掃描 spatial_freq
    print("\n--- Phase 2: spatial_freq ---")
    spatial_freqs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results_phase2 = []

    print(f"\n{'sp_freq':>7s} {'active':>7s} {'DSI_med':>8s} {'DSI>0.6':>8s}")
    print("-" * 35)

    for sf in spatial_freqs:
        result = run_single(
            brain, mi1_indices, mi1_xy, t4_t5_result, directions,
            duration_ms=300, mean_drive=best_phase1["drive"],
            spatial_freq=sf, temporal_freq=best_phase1["freq"]
        )
        results_phase2.append({"spatial_freq": sf, **result})
        print(f"{sf:7.3f} {result['n_active']:7d} "
              f"{result['dsi_median_active']:8.3f} {result['dsi_gt_06_pct']:7.1f}%")

    valid2 = [r for r in results_phase2 if r["n_active"] >= 10]
    best_phase2 = max(valid2, key=lambda r: r["dsi_median_active"])
    print(f"\n✅ Phase 2 最佳: spatial_freq={best_phase2['spatial_freq']}")
    print(f"   DSI median = {best_phase2['dsi_median_active']:.3f}")

    # Phase 3: 用最佳參數跑不同 duration
    print("\n--- Phase 3: duration_ms ---")
    durations = [300, 500, 1000, 2000]
    results_phase3 = []

    best_params = {
        "mean_drive": best_phase1["drive"],
        "temporal_freq": best_phase1["freq"],
        "spatial_freq": best_phase2["spatial_freq"],
    }

    print(f"\n用最佳參數: drive={best_params['mean_drive']}, "
          f"freq={best_params['temporal_freq']}, sp_freq={best_params['spatial_freq']}")
    print(f"\n{'dur_ms':>7s} {'active':>7s} {'DSI_med':>8s} {'DSI>0.6':>8s}")
    print("-" * 35)

    for dur in durations:
        result = run_single(
            brain, mi1_indices, mi1_xy, t4_t5_result, directions,
            duration_ms=dur, **best_params
        )
        results_phase3.append({"duration_ms": dur, **result})
        print(f"{dur:7d} {result['n_active']:7d} "
              f"{result['dsi_median_active']:8.3f} {result['dsi_gt_06_pct']:7.1f}%")

    valid3 = [r for r in results_phase3 if r["n_active"] >= 10]
    best_phase3 = max(valid3, key=lambda r: r["dsi_median_active"])
    best_params["duration_ms"] = best_phase3["duration_ms"]

    print(f"\n✅ Phase 3 最佳: duration_ms={best_phase3['duration_ms']}")
    print(f"   DSI median = {best_phase3['dsi_median_active']:.3f}")

    # 最終結果
    print("\n" + "=" * 70)
    print("最終最佳參數")
    print("=" * 70)
    print(json.dumps(best_params, indent=2))
    print(f"\n最佳 DSI median: {best_phase3['dsi_median_active']:.3f}")
    print(f"最佳 DSI > 0.6 比例: {best_phase3['dsi_gt_06_pct']:.1f}%")
    print(f"active cells: {best_phase3['n_active']}")

    # 對照 Maisak 2013
    print(f"\n--- 對照 Maisak 2013 (0.6-0.8) ---")
    if 0.6 <= best_phase3['dsi_median_active'] <= 0.8:
        print("✅ 落在生物學區間!")
    elif best_phase3['dsi_median_active'] > 0.8:
        print("⚠️ 高於生物學區間(可能過強刺激)")
    else:
        print(f"⚠️ 低於生物學區間({best_phase3['dsi_median_active']:.3f} < 0.6)")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "param_sweep_results.json", "w") as f:
        json.dump({
            "phase1": results_phase1,
            "phase2": results_phase2,
            "phase3": results_phase3,
            "best_params": best_params,
            "best_dsi_median": best_phase3['dsi_median_active'],
        }, f, indent=2)
    print(f"\n💾 結果已儲存至 {output_path}/param_sweep_results.json")


if __name__ == "__main__":
    main()
