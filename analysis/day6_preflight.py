"""MaleCNS 原版 drifting grating 實驗 — 第 6 天預檢 v2

修正:v1 用 seq.mean(axis=0) 把時間維度平均掉,導致 grating 空間結構消失。
v2 逐 batch 餵入時變 luminance,讓光感受器收到真正的 drifting grating。

核心問題:凍結 MaleCNS 連接體 + LIF 動力學 + drifting grating,
能否在 T4/T5 細胞上重現 Maisak et al. 2013 (Nature) 的方向選擇性?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# 設定路徑 — repo root 在 path[0],這樣 neural 和 stimuli 都是 package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural.common import DATA, GRAPH
from neural.t4_t5 import identify_t4_t5
from neural.brain import MemoryBrain
from stimuli.grating import GratingStimulus


def compute_dsi(spike_rates, directions):
    """計算方向選擇性指數 DSI = |R_pref - R_null| / (R_pref + R_null)"""
    n_cells, n_dirs = spike_rates.shape
    dsi = np.zeros(n_cells, dtype=np.float32)
    preferred_dir = np.zeros(n_cells, dtype=np.int32)
    null_dir = np.zeros(n_cells, dtype=np.int32)

    for i in range(n_cells):
        rates = spike_rates[i]
        pref_idx = np.argmax(rates)
        preferred_dir[i] = directions[pref_idx]
        null_idx = (pref_idx + n_dirs // 2) % n_dirs
        null_dir[i] = directions[null_idx]

        r_pref = rates[pref_idx]
        r_null = rates[null_idx]
        denom = r_pref + r_null
        if denom > 0:
            dsi[i] = abs(r_pref - r_null) / denom
        else:
            dsi[i] = 0.0

    return dsi, preferred_dir, null_dir


def run_direction(brain, stimulus, direction_deg, t4_t5_indices, duration_ms=2000,
                  batch_ms=10):
    """跑單一方向的 grating,逐 batch 餵入時變 luminance

    Parameters
    ----------
    brain : MemoryBrain
    stimulus : GratingStimulus
    direction_deg : float
    t4_t5_indices : array
    duration_ms : float
    batch_ms : float
        每批時長(ms),luminance 在該批次內平均

    Returns
    -------
    t4_t5_spike_counts : array (n_t4_t5,)
    """
    seq, times = stimulus.generate_direction(direction_deg, duration_ms)
    # seq shape: (T, N_receptors), T = duration_ms / dt
    dt = stimulus.DT
    batch_steps = int(round(batch_ms / dt))

    n_t4_t5 = len(t4_t5_indices)
    total_spikes = np.zeros(n_t4_t5, dtype=np.int32)

    n_batches = len(seq) // batch_steps
    t_start = time.time()

    for b in range(n_batches):
        # 取這個 batch 的平均 luminance(保留空間結構,只平均時間)
        batch = seq[b * batch_steps : (b + 1) * batch_steps]
        mean_lum = batch.mean(axis=0)  # (N_receptors,)

        # 跑這個 batch
        counts, _ = brain.step(mean_lum, batch_ms, learning=False)
        total_spikes += counts[t4_t5_indices]

        if b % 20 == 0:  # 每 200ms 報告
            elapsed = time.time() - t_start
            progress = (b + 1) / n_batches
            print(f"    {progress*100:.1f}% ({(b+1)*batch_ms:.0f}ms, {elapsed:.1f}s)",
                  flush=True)

    return total_spikes


def main():
    print("=" * 70)
    print("MaleCNS 原版 Drifting Grating 實驗 v2")
    print("第 6 天預檢: T4/T5 DSI vs Maisak 2013 生物學區間")
    print("=" * 70)

    # 載入
    print("\n--- 載入 ---")
    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    t4 = t4_t5_result["t4"]
    t5 = t4_t5_result["t5"]
    t4_t5 = t4_t5_result["t4_t5"]
    print(f"T4: {len(t4)}, T5: {len(t5)}")

    from neural.brain import MemoryBrain
    brain = MemoryBrain(GRAPH, eta=0.0)
    print(f"Brain: {brain.n} neurons")

    stimulus = GratingStimulus(GRAPH)
    print(f"Stimulus: {stimulus.n_receptors} receptors")

    # Pipeline 測試
    print("\n--- Pipeline 測試: 500ms 0° grating (逐 batch) ---")
    brain.reset(keep_memory=False)
    test_counts = run_direction(brain, stimulus, 0, t4_t5, duration_ms=500, batch_ms=10)
    print(f"  T4/T5 spike (500ms): {test_counts.sum()}")
    if test_counts.sum() == 0:
        print("  ⚠️ T4/T5 仍無 spike,診斷中...")
        # 診斷
        brain.reset(keep_memory=False)
        seq, _ = stimulus.generate_direction(0, duration_ms=500)
        batch = seq[:100]  # 第一個 10ms
        mean_lum = batch.mean(axis=0)
        print(f"  luminance 範圍: [{mean_lum.min():.3f}, {mean_lum.max():.3f}]")
        counts, _ = brain.step(mean_lum, 10, learning=False)
        print(f"  R1-R6 spike: {counts[brain.retina].sum()}")
        print(f"  L1 spike: {counts[t4_t5_result['upstream']['L1']].sum()}")
        print(f"  L2 spike: {counts[t4_t5_result['upstream']['L2']].sum()}")
        print(f"  L3 spike: {counts[t4_t5_result['upstream']['L3']].sum()}")
        print(f"  Mi1 spike: {counts[t4_t5_result['upstream']['Mi1']].sum()}")
        print(f"  Tm3 spike: {counts[t4_t5_result['upstream']['Tm3']].sum()}")
        print(f"  T4 膜電位: {brain.v[t4].mean():.2f} mV (閾值 -45)")
        print(f"  T5 膜電位: {brain.v[t5].mean():.2f} mV (閾值 -45)")
        print(f"  T4 drive: {brain.drive[t4].mean():.2f}")
        print(f"  T5 drive: {brain.drive[t5].mean():.2f}")

    # 完整實驗
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    duration_ms = 2000
    n_t4_t5 = len(t4_t5)
    n_dirs = len(directions)
    spike_counts = np.zeros((n_t4_t5, n_dirs), dtype=np.int32)

    print(f"\n--- 完整實驗: {n_dirs} 方向 × {duration_ms}ms ---")
    for d_idx, direction in enumerate(directions):
        print(f"\n  方向 {direction}° ({d_idx+1}/{n_dirs}):")
        brain.reset(keep_memory=False)
        counts = run_direction(brain, stimulus, direction, t4_t5,
                                duration_ms=duration_ms, batch_ms=10)
        spike_counts[:, d_idx] = counts
        duration_s = duration_ms / 1000.0
        rates = counts / duration_s
        print(f"    T4 平均激發率: {rates[:len(t4)].mean():.2f} Hz")
        print(f"    T5 平均激發率: {rates[len(t4):].mean():.2f} Hz")

    # 計算 DSI
    duration_s = duration_ms / 1000.0
    spike_rates = spike_counts / duration_s
    dsi, pref_dir, null_dir = compute_dsi(spike_rates, directions)

    # 報告
    print("\n" + "=" * 70)
    print("DSI 分析結果")
    print("=" * 70)

    n_t4 = len(t4)
    n_t5 = len(t5)
    t4_dsi = dsi[:n_t4]
    t5_dsi = dsi[n_t4:]
    t4_rates = spike_rates[:n_t4]
    t5_rates = spike_rates[n_t4:]

    print(f"\n--- T4 細胞 (n={n_t4}) ---")
    print(f"  DSI 中位數: {np.median(t4_dsi):.3f}")
    print(f"  DSI 平均: {t4_dsi.mean():.3f}")
    print(f"  DSI > 0.3 比例: {(t4_dsi > 0.3).mean()*100:.1f}%")
    print(f"  DSI > 0.6 比例: {(t4_dsi > 0.6).mean()*100:.1f}%")
    print(f"  平均激發率: {t4_rates.mean():.2f} Hz")
    print(f"  最大激發率: {t4_rates.max():.2f} Hz")

    print(f"\n--- T5 細胞 (n={n_t5}) ---")
    print(f"  DSI 中位數: {np.median(t5_dsi):.3f}")
    print(f"  DSI 平均: {t5_dsi.mean():.3f}")
    print(f"  DSI > 0.3 比例: {(t5_dsi > 0.3).mean()*100:.1f}%")
    print(f"  DSI > 0.6 比例: {(t5_dsi > 0.6).mean()*100:.1f}%")
    print(f"  平均激發率: {t5_rates.mean():.2f} Hz")
    print(f"  最大激發率: {t5_rates.max():.2f} Hz")

    print(f"\n--- 對照 Maisak 2013 (Nature) ---")
    print(f"  真實 T4/T5 DSI: 0.6-0.8 (GCaMP 電生理)")
    print(f"  模擬 T4 DSI 中位數: {np.median(t4_dsi):.3f}")
    print(f"  模擬 T5 DSI 中位數: {np.median(t5_dsi):.3f}")

    t4_in_range = 0.6 <= np.median(t4_dsi) <= 0.8
    t5_in_range = 0.6 <= np.median(t5_dsi) <= 0.8

    print(f"\n{'='*70}")
    print("GO / NO-GO DECISION")
    print(f"{'='*70}")
    if t4_in_range or t5_in_range:
        print("✅ GO: T4/T5 DSI 落在生物學區間,pipeline 驗證成功")
    elif np.median(t4_dsi) > 0.3 or np.median(t5_dsi) > 0.3:
        print("⚠️  PARTIAL: 有方向選擇性但不夠強")
        print(f"   T4 DSI: {np.median(t4_dsi):.3f}, T5 DSI: {np.median(t5_dsi):.3f}")
    elif t4_rates.mean() > 0 or t5_rates.mean() > 0:
        print("⚠️  T4/T5 有 spike 但 DSI 低")
        print(f"   T4 DSI: {np.median(t4_dsi):.3f}, T5 DSI: {np.median(t5_dsi):.3f}")
        print("   → 可能需要更長時長或更強刺激")
    else:
        print("❌ NO-GO: T4/T5 無 spike")
        print("   → pipeline 有問題,需要 debug")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / "day6_dsi_results.npz",
        t4_dsi=t4_dsi, t5_dsi=t5_dsi,
        t4_rates=t4_rates, t5_rates=t5_rates,
        t4_pref_dir=pref_dir[:n_t4], t5_pref_dir=pref_dir[n_t4:],
        directions=directions, spike_counts=spike_counts,
    )
    report = {
        "experiment": "MaleCNS frozen drifting grating v2",
        "n_t4": n_t4, "n_t5": n_t5,
        "directions": directions,
        "duration_per_direction_ms": duration_ms,
        "t4_dsi_median": float(np.median(t4_dsi)),
        "t4_dsi_mean": float(t4_dsi.mean()),
        "t4_dsi_gt_0_3_pct": float((t4_dsi > 0.3).mean() * 100),
        "t4_dsi_gt_0_6_pct": float((t4_dsi > 0.6).mean() * 100),
        "t4_mean_rate_hz": float(t4_rates.mean()),
        "t5_dsi_median": float(np.median(t5_dsi)),
        "t5_dsi_mean": float(t5_dsi.mean()),
        "t5_dsi_gt_0_3_pct": float((t5_dsi > 0.3).mean() * 100),
        "t5_dsi_gt_0_6_pct": float((t5_dsi > 0.6).mean() * 100),
        "t5_mean_rate_hz": float(t5_rates.mean()),
        "biological_reference": "Maisak et al. 2013, Nature, DSI 0.6-0.8",
        "t4_in_biological_range": bool(t4_in_range),
        "t5_in_biological_range": bool(t5_in_range),
    }
    (output_path / "day6_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n💾 結果已儲存至 {output_path}/")


if __name__ == "__main__":
    main()
