"""Null model 對照實驗

核心問題:MaleCNS 連接體的方向選擇性,是來自「具體接線」還是「度分布」還是「純隨機也行」?

三組:
1. MaleCNS 原版(實驗組)
2. Configuration model:每個節點保留 out-degree,但目標隨機重連
3. Erdős–Rényi:完全隨機 post,只保留邊數

每組用相同 HR 刺激(方向 C),比較 T4 DSI 分布。
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
    mi1_indices = []
    mi1_hex = []
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


def moving_wave_stimulus(mi1_indices, mi1_xy, direction_deg, t_ms,
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


def apply_configuration_model(brain, seed):
    """Configuration model: shuffle post within each node's outgoing edges

    保留每個節點的 out-degree 和 weight,但目標隨機重連。
    """
    rng = np.random.RandomState(seed)
    post = brain.post
    ptr = brain.ptr
    for i in range(brain.n):
        start, end = ptr[i], ptr[i + 1]
        if end > start:
            rng.shuffle(post[start:end])
    return brain


def apply_er_model(brain, seed):
    """Erdős–Rényi: completely random post, same edge count"""
    rng = np.random.RandomState(seed)
    brain.post[:] = rng.randint(0, brain.n, size=len(brain.post), dtype=np.int32)
    return brain


def run_experiment(brain, mi1_indices, mi1_xy, t4_t5_result,
                    directions, duration_ms=300, mean_drive=12.0):
    """跑 8 方向 HR 刺激,返回 T4/T5 DSI"""
    t4 = t4_t5_result["t4"]
    t5 = t4_t5_result["t5"]
    tonic = np.zeros(brain.n, dtype=np.float32)
    tonic[t4_t5_result["upstream"]["Mi4"]] = 3.0
    brain.tonic = tonic

    dummy_lum = np.zeros(len(brain.retina), dtype=np.float32)
    n_batches = int(duration_ms / 10)

    t4_rates_all = []
    t5_rates_all = []

    for direction in directions:
        brain.reset(keep_memory=False)
        t4_counts = np.zeros(len(t4), dtype=np.int32)
        t5_counts = np.zeros(len(t5), dtype=np.int32)

        for b in range(n_batches):
            t_ms = b * 10
            mi1_drive = moving_wave_stimulus(
                mi1_indices, mi1_xy, direction, t_ms, mean_drive=mean_drive
            )
            counts, _ = brain.step(
                dummy_lum, 10, learning=False,
                stimulation=[(mi1_indices, mi1_drive)]
            )
            t4_counts += counts[t4]
            t5_counts += counts[t5]

        t4_rate = t4_counts / (duration_ms / 1000)
        t5_rate = t5_counts / (duration_ms / 1000)
        t4_rates_all.append(t4_rate)
        t5_rates_all.append(t5_rate)

    t4_rates_all = np.array(t4_rates_all)
    t5_rates_all = np.array(t5_rates_all)

    def compute_dsi(rates_matrix, directions):
        n_dirs, n_cells = rates_matrix.shape
        dsi = np.zeros(n_cells, dtype=np.float32)
        pref_dir = np.zeros(n_cells, dtype=np.int32)
        for i in range(n_cells):
            rates = rates_matrix[:, i]
            if rates.sum() == 0:
                continue
            pref_idx = np.argmax(rates)
            null_idx = (pref_idx + 4) % 8
            r_pref = rates[pref_idx]
            r_null = rates[null_idx]
            denom = r_pref + r_null
            if denom > 0:
                dsi[i] = abs(r_pref - r_null) / denom
            pref_dir[i] = directions[pref_idx]
        return dsi, pref_dir

    t4_dsi, t4_pref = compute_dsi(t4_rates_all, directions)
    t5_dsi, t5_pref = compute_dsi(t5_rates_all, directions)

    t4_active = t4_rates_all.sum(axis=0) > 0
    t5_active = t5_rates_all.sum(axis=0) > 0

    return {
        "t4_dsi": t4_dsi, "t5_dsi": t5_dsi,
        "t4_pref": t4_pref, "t5_pref": t5_pref,
        "t4_active": t4_active, "t5_active": t5_active,
        "t4_rates": t4_rates_all, "t5_rates": t5_rates_all,
    }


def summarize(result, name):
    t4_dsi = result["t4_dsi"]
    t4_active = result["t4_active"]
    t5_dsi = result["t5_dsi"]
    t5_active = result["t5_active"]

    n_t4_active = t4_active.sum()
    n_t5_active = t5_active.sum()

    t4_dsi_active = t4_dsi[t4_active] if n_t4_active > 0 else np.array([])
    t5_dsi_active = t5_dsi[t5_active] if n_t5_active > 0 else np.array([])

    return {
        "name": name,
        "t4_active_cells": int(n_t4_active),
        "t4_dsi_median_all": float(np.median(t4_dsi)),
        "t4_dsi_median_active": float(np.median(t4_dsi_active)) if len(t4_dsi_active) > 0 else 0,
        "t4_dsi_gt_03_pct_active": float((t4_dsi_active > 0.3).mean() * 100) if len(t4_dsi_active) > 0 else 0,
        "t4_dsi_gt_06_pct_active": float((t4_dsi_active > 0.6).mean() * 100) if len(t4_dsi_active) > 0 else 0,
        "t5_active_cells": int(n_t5_active),
        "t5_dsi_median_active": float(np.median(t5_dsi_active)) if len(t5_dsi_active) > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Null Model 對照實驗")
    print("MaleCNS 原版 vs Configuration model vs Erdős–Rényi")
    print("=" * 70)

    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    mi1_indices, mi1_xy = get_mi1_spatial_layout(ids, t4_t5_result)
    print(f"\nMi1 有座標: {len(mi1_indices)}")

    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    duration_ms = 300
    mean_drive = 12.0
    n_seeds = 3

    all_results = {}

    # 1. MaleCNS 原版
    print("\n" + "=" * 50)
    print("1. MaleCNS 原版")
    print("=" * 50)
    brain = MemoryBrain(GRAPH, eta=0.0)
    result = run_experiment(brain, mi1_indices, mi1_xy, t4_t5_result,
                             directions, duration_ms, mean_drive)
    summary_orig = summarize(result, "MaleCNS 原版")
    print(f"  T4 active: {summary_orig['t4_active_cells']}")
    print(f"  T4 DSI median (active): {summary_orig['t4_dsi_median_active']:.3f}")
    print(f"  T4 DSI > 0.6: {summary_orig['t4_dsi_gt_06_pct_active']:.1f}%")
    all_results["malecns_original"] = result

    # 2. Configuration model (5 seeds)
    print("\n" + "=" * 50)
    print("2. Configuration model (保留 out-degree,隨機重連)")
    print("=" * 50)
    config_summaries = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}:")
        brain = MemoryBrain(GRAPH, eta=0.0)
        apply_configuration_model(brain, seed)
        result = run_experiment(brain, mi1_indices, mi1_xy, t4_t5_result,
                                 directions, duration_ms, mean_drive)
        s = summarize(result, f"Config seed {seed}")
        print(f"    T4 active: {s['t4_active_cells']}, DSI median: {s['t4_dsi_median_active']:.3f}")
        config_summaries.append(s)
        all_results[f"config_seed{seed}"] = result

    # 3. Erdős–Rényi model (5 seeds)
    print("\n" + "=" * 50)
    print("3. Erdős–Rényi model (完全隨機)")
    print("=" * 50)
    er_summaries = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}:")
        brain = MemoryBrain(GRAPH, eta=0.0)
        apply_er_model(brain, seed)
        result = run_experiment(brain, mi1_indices, mi1_xy, t4_t5_result,
                                 directions, duration_ms, mean_drive)
        s = summarize(result, f"ER seed {seed}")
        print(f"    T4 active: {s['t4_active_cells']}, DSI median: {s['t4_dsi_median_active']:.3f}")
        er_summaries.append(s)
        all_results[f"er_seed{seed}"] = result

    # 彙總比較
    print("\n" + "=" * 70)
    print("彙總比較")
    print("=" * 70)

    print(f"\n{'組別':25s} {'T4 active':>10s} {'DSI med':>8s} {'DSI>0.3':>8s} {'DSI>0.6':>8s}")
    print("-" * 65)

    # MaleCNS 原版
    s = summary_orig
    print(f"{'MaleCNS 原版':25s} {s['t4_active_cells']:10d} {s['t4_dsi_median_active']:8.3f} "
          f"{s['t4_dsi_gt_03_pct_active']:7.1f}% {s['t4_dsi_gt_06_pct_active']:7.1f}%")

    # Configuration model
    config_actives = [s['t4_active_cells'] for s in config_summaries]
    config_dsis = [s['t4_dsi_median_active'] for s in config_summaries]
    config_03 = [s['t4_dsi_gt_03_pct_active'] for s in config_summaries]
    config_06 = [s['t4_dsi_gt_06_pct_active'] for s in config_summaries]
    print(f"{'Config (mean±std)':25s} {np.mean(config_actives):10.1f} {np.mean(config_dsis):8.3f} "
          f"{np.mean(config_03):7.1f}% {np.mean(config_06):7.1f}%")
    print(f"{'Config (std)':25s} {np.std(config_actives):10.1f} {np.std(config_dsis):8.3f} "
          f"{np.std(config_03):7.1f}% {np.std(config_06):7.1f}%")

    # ER model
    er_actives = [s['t4_active_cells'] for s in er_summaries]
    er_dsis = [s['t4_dsi_median_active'] for s in er_summaries]
    er_03 = [s['t4_dsi_gt_03_pct_active'] for s in er_summaries]
    er_06 = [s['t4_dsi_gt_06_pct_active'] for s in er_summaries]
    print(f"{'ER (mean±std)':25s} {np.mean(er_actives):10.1f} {np.mean(er_dsis):8.3f} "
          f"{np.mean(er_03):7.1f}% {np.mean(er_06):7.1f}%")
    print(f"{'ER (std)':25s} {np.std(er_actives):10.1f} {np.std(er_dsis):8.3f} "
          f"{np.std(er_03):7.1f}% {np.std(er_06):7.1f}%")

    # 結論
    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)

    orig_dsi = summary_orig['t4_dsi_median_active']
    config_dsi_mean = np.mean(config_dsis)
    er_dsi_mean = np.mean(er_dsis)

    print(f"\nMaleCNS 原版 DSI: {orig_dsi:.3f}")
    print(f"Configuration model DSI: {config_dsi_mean:.3f} ± {np.std(config_dsis):.3f}")
    print(f"Erdős–Rényi DSI: {er_dsi_mean:.3f} ± {np.std(er_dsis):.3f}")

    if orig_dsi > config_dsi_mean + 2 * np.std(config_dsis) and config_dsi_mean > er_dsi_mean:
        print("\n✅ 結果 A: MaleCNS > Configuration > ER")
        print("   連接體的「具體接線」和「度分布」都對方向選擇性有先驗貢獻")
        print("   → 殺手級結果,可投 Nature/Science")
    elif abs(orig_dsi - config_dsi_mean) < np.std(config_dsis) and config_dsi_mean > er_dsi_mean:
        print("\n✅ 結果 B: MaleCNS = Configuration > ER")
        print("   方向選擇性來自「度分布」,不是「具體接線」")
        print("   → 重要結果,可投 NeurIPS/PNAS")
    elif abs(orig_dsi - config_dsi_mean) < np.std(config_dsis) and abs(config_dsi_mean - er_dsi_mean) < np.std(er_dsis):
        print("\n✅ 結果 C: MaleCNS = Configuration = ER")
        print("   連接體拓撲不提供方向選擇性先驗")
        print("   → 負面結果,推翻「連接體即先驗」假設")
    else:
        print("\n⚠️  混合結果,需要進一步分析")
        print(f"   MaleCNS={orig_dsi:.3f}, Config={config_dsi_mean:.3f}±{np.std(config_dsis):.3f}, ER={er_dsi_mean:.3f}±{np.std(er_dsis):.3f}")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    output_path.mkdir(parents=True, exist_ok=True)

    # 儲存 DSI 分布
    np.savez(
        output_path / "null_model_comparison.npz",
        # MaleCNS 原版
        orig_t4_dsi=all_results["malecns_original"]["t4_dsi"],
        orig_t4_active=all_results["malecns_original"]["t4_active"],
        # Configuration
        config_t4_dsis=np.array([all_results[f"config_seed{s}"]["t4_dsi"] for s in range(n_seeds)]),
        config_t4_actives=np.array([all_results[f"config_seed{s}"]["t4_active"] for s in range(n_seeds)]),
        # ER
        er_t4_dsis=np.array([all_results[f"er_seed{s}"]["t4_dsi"] for s in range(n_seeds)]),
        er_t4_actives=np.array([all_results[f"er_seed{s}"]["t4_active"] for s in range(n_seeds)]),
        directions=directions,
    )

    # 儲存報告
    report = {
        "experiment": "Null model comparison",
        "malecns_original": summary_orig,
        "configuration_model": {
            "seeds": n_seeds,
            "summaries": config_summaries,
            "t4_dsi_mean": float(np.mean(config_dsis)),
            "t4_dsi_std": float(np.std(config_dsis)),
            "t4_active_mean": float(np.mean(config_actives)),
        },
        "er_model": {
            "seeds": n_seeds,
            "summaries": er_summaries,
            "t4_dsi_mean": float(np.mean(er_dsis)),
            "t4_dsi_std": float(np.std(er_dsis)),
            "t4_active_mean": float(np.mean(er_actives)),
        },
    }
    (output_path / "null_model_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n💾 結果已儲存至 {output_path}/")


if __name__ == "__main__":
    main()
