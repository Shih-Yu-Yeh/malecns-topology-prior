"""Null model 對照實驗 v2 — 用參數掃描最佳值

用 mean_drive=15, temporal_freq=1, spatial_freq=0.05, duration=300ms
預期 MaleCNS 有數百 active cells,Config/ER 應該顯著更低。
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


def moving_wave(mi1_xy, direction_deg, t_ms, sf=0.05, tf=1.0, drive=15.0):
    rad = np.deg2rad(direction_deg)
    proj = mi1_xy[:, 0] * np.cos(rad) + mi1_xy[:, 1] * np.sin(rad)
    phase = 2 * np.pi * tf * (t_ms / 1000.0)
    return np.clip(drive + drive * np.sin(2 * np.pi * sf * proj - phase), 0, None).astype(np.float32)


def apply_config_model(brain, seed):
    rng = np.random.RandomState(seed)
    for i in range(brain.n):
        s, e = brain.ptr[i], brain.ptr[i + 1]
        if e > s:
            rng.shuffle(brain.post[s:e])
    return brain


def apply_er_model(brain, seed):
    rng = np.random.RandomState(seed)
    brain.post[:] = rng.randint(0, brain.n, size=len(brain.post), dtype=np.int32)
    return brain


def run_8dir(brain, mi1_indices, mi1_xy, t4_t5_result,
              duration_ms=300, drive=15.0, sf=0.05, tf=1.0):
    t4 = t4_t5_result["t4"]
    tonic = np.zeros(brain.n, dtype=np.float32)
    tonic[t4_t5_result["upstream"]["Mi4"]] = 3.0
    brain.tonic = tonic
    dummy_lum = np.zeros(len(brain.retina), dtype=np.float32)
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    n_batches = int(duration_ms / 10)
    rates_all = []
    for direction in directions:
        brain.reset(keep_memory=False)
        counts = np.zeros(len(t4), dtype=np.int32)
        for b in range(n_batches):
            mi1_drive = moving_wave(mi1_xy, direction, b * 10, sf, tf, drive)
            c, _ = brain.step(dummy_lum, 10, learning=False,
                               stimulation=[(mi1_indices, mi1_drive)])
            counts += c[t4]
        rates_all.append(counts / (duration_ms / 1000))
    rates = np.array(rates_all)
    dsi = np.zeros(len(t4), dtype=np.float32)
    pref = np.zeros(len(t4), dtype=np.int32)
    for i in range(len(t4)):
        r = rates[:, i]
        if r.sum() == 0:
            continue
        pi = np.argmax(r)
        ni = (pi + 4) % 8
        denom = r[pi] + r[ni]
        if denom > 0:
            dsi[i] = abs(r[pi] - r[ni]) / denom
        pref[i] = directions[pi]
    active = rates.sum(axis=0) > 0
    return dsi, pref, active, rates


def summarize(dsi, active):
    n = active.sum()
    dsi_a = dsi[active] if n > 0 else np.array([])
    return {
        "n_active": int(n),
        "dsi_median": float(np.median(dsi_a)) if len(dsi_a) > 0 else 0,
        "dsi_mean": float(dsi_a.mean()) if len(dsi_a) > 0 else 0,
        "dsi_gt_03_pct": float((dsi_a > 0.3).mean() * 100) if len(dsi_a) > 0 else 0,
        "dsi_gt_06_pct": float((dsi_a > 0.6).mean() * 100) if len(dsi_a) > 0 else 0,
        "dsi_gt_08_pct": float((dsi_a > 0.8).mean() * 100) if len(dsi_a) > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Null Model 對照 v2 — 最佳參數 (drive=15, freq=1)")
    print("=" * 70)

    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    mi1_indices, mi1_xy = get_mi1_spatial_layout(ids, t4_t5_result)

    # 最佳參數
    params = {"drive": 15.0, "sf": 0.05, "tf": 1.0, "duration_ms": 300}
    n_seeds = 5

    # 1. MaleCNS 原版
    print("\n--- 1. MaleCNS 原版 ---")
    brain = MemoryBrain(GRAPH, eta=0.0)
    t0 = time.time()
    dsi, pref, active, rates = run_8dir(brain, mi1_indices, mi1_xy, t4_t5_result, **params)
    s_orig = summarize(dsi, active)
    print(f"  ({time.time()-t0:.0f}s)")
    print(f"  Active: {s_orig['n_active']}")
    print(f"  DSI median: {s_orig['dsi_median']:.3f}")
    print(f"  DSI > 0.6: {s_orig['dsi_gt_06_pct']:.1f}%")

    # 2. Configuration model
    print("\n--- 2. Configuration model (5 seeds) ---")
    config_results = []
    for seed in range(n_seeds):
        brain = MemoryBrain(GRAPH, eta=0.0)
        apply_config_model(brain, seed)
        t0 = time.time()
        dsi, pref, active, rates = run_8dir(brain, mi1_indices, mi1_xy, t4_t5_result, **params)
        s = summarize(dsi, active)
        config_results.append(s)
        print(f"  Seed {seed}: active={s['n_active']}, DSI={s['dsi_median']:.3f} ({time.time()-t0:.0f}s)")

    # 3. ER model
    print("\n--- 3. Erdős–Rényi (5 seeds) ---")
    er_results = []
    for seed in range(n_seeds):
        brain = MemoryBrain(GRAPH, eta=0.0)
        apply_er_model(brain, seed)
        t0 = time.time()
        dsi, pref, active, rates = run_8dir(brain, mi1_indices, mi1_xy, t4_t5_result, **params)
        s = summarize(dsi, active)
        er_results.append(s)
        print(f"  Seed {seed}: active={s['n_active']}, DSI={s['dsi_median']:.3f} ({time.time()-t0:.0f}s)")

    # 彙總
    print("\n" + "=" * 70)
    print("彙總")
    print("=" * 70)
    print(f"\n{'組別':20s} {'Active':>8s} {'DSI med':>8s} {'DSI>0.3':>8s} {'DSI>0.6':>8s}")
    print("-" * 60)
    print(f"{'MaleCNS 原版':20s} {s_orig['n_active']:8d} {s_orig['dsi_median']:8.3f} "
          f"{s_orig['dsi_gt_03_pct']:7.1f}% {s_orig['dsi_gt_06_pct']:7.1f}%")
    
    c_act = [s['n_active'] for s in config_results]
    c_dsi = [s['dsi_median'] for s in config_results]
    c_03 = [s['dsi_gt_03_pct'] for s in config_results]
    c_06 = [s['dsi_gt_06_pct'] for s in config_results]
    print(f"{'Config (mean±std)':20s} {np.mean(c_act):8.0f}±{np.std(c_act):4.0f} "
          f"{np.mean(c_dsi):8.3f}±{np.std(c_dsi):.3f} {np.mean(c_03):7.1f}% {np.mean(c_06):7.1f}%")
    
    e_act = [s['n_active'] for s in er_results]
    e_dsi = [s['dsi_median'] for s in er_results]
    e_03 = [s['dsi_gt_03_pct'] for s in er_results]
    e_06 = [s['dsi_gt_06_pct'] for s in er_results]
    print(f"{'ER (mean±std)':20s} {np.mean(e_act):8.0f}±{np.std(e_act):4.0f} "
          f"{np.mean(e_dsi):8.3f}±{np.std(e_dsi):.3f} {np.mean(e_03):7.1f}% {np.mean(e_06):7.1f}%")

    # 統計檢定
    from scipy import stats as scipy_stats
    try:
        # MaleCNS vs Config (one-sample t-test, MaleCNS vs Config mean)
        t_stat, p_val = scipy_stats.ttest_1samp(c_dsi, s_orig['dsi_median'])
        print(f"\n--- 統計檢定 ---")
        print(f"MaleCNS DSI = {s_orig['dsi_median']:.3f}")
        print(f"Config DSI = {np.mean(c_dsi):.3f} ± {np.std(c_dsi):.3f}")
        print(f"Config vs MaleCNS: t={t_stat:.2f}, p={p_val:.6f}")
        
        t_stat2, p_val2 = scipy_stats.ttest_ind(c_dsi, e_dsi)
        print(f"Config vs ER: t={t_stat2:.2f}, p={p_val2:.6f}")
    except:
        print(f"\n(scipy 不可用,跳過統計檢定)")

    # 結論
    print(f"\n{'='*70}")
    print("結論")
    print(f"{'='*70}")
    
    orig_dsi = s_orig['dsi_median']
    config_mean = np.mean(c_dsi)
    config_std = np.std(c_dsi)
    er_mean = np.mean(e_dsi)
    er_std = np.std(e_dsi)
    
    print(f"\nMaleCNS: {orig_dsi:.3f}")
    print(f"Config:  {config_mean:.3f} ± {config_std:.3f}")
    print(f"ER:      {er_mean:.3f} ± {er_std:.3f}")
    
    orig_gt_config = orig_dsi > config_mean + 2 * config_std
    config_gt_er = config_mean > er_mean + 2 * er_std
    
    if orig_gt_config and config_gt_er:
        print("\n✅✅ 結果 A (強版): MaleCNS >> Config >> ER")
        print("   具體接線 + 度分布都有顯著先驗貢獻")
        print("   → 殺手級結果")
    elif orig_gt_config:
        print("\n✅ 結果 A (弱版): MaleCNS >> Config ≈ ER")
        print("   具體接線有先驗貢獻,度分布無顯著貢獻")
    else:
        print("\n⚠️ 需要更多分析")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    report = {
        "params": params,
        "malecns": s_orig,
        "config": {"seeds": n_seeds, "results": config_results,
                    "dsi_mean": float(np.mean(c_dsi)), "dsi_std": float(np.std(c_dsi))},
        "er": {"seeds": n_seeds, "results": er_results,
               "dsi_mean": float(np.mean(e_dsi)), "dsi_std": float(np.std(e_dsi))},
    }
    (output_path / "null_model_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n💾 儲存至 {output_path}/null_model_v2_report.json")


if __name__ == "__main__":
    main()
