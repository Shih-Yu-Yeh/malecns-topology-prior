"""MaleCNS 原版 drifting grating 實驗 — Day 6 v3

修正:v2 發現 Mi1/Tm3 膜電位被抑制到 -55mV,無法過閾值。
v3 給 medulla 細胞(Mi1/Tm3/Mi4/Mi9/Tm9)tonic drive,把它們推到接近閾值。

這是「方向 A:給 medulla 細胞 tonic drive」的實作。
tonic drive 是非生物學的,但目的是證明訊號能傳到 T4/T5,不是模擬真實果蠅。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural.common import DATA, GRAPH
from neural.t4_t5 import identify_t4_t5
from stimuli.grating import GratingStimulus


def build_tonic(brain, t4_t5_result, medulla_tonic=3.0):
    """建立 tonic drive 陣列

    給 medulla 細胞(Mi1/Tm3/Mi4/Mi9/Tm9)tonic drive,
    把它們的靜息電位從 -52 推到 -52 + tonic。

    Parameters
    ----------
    brain : MemoryBrain
    t4_t5_result : dict
    medulla_tonic : float
        tonic drive 值(mV 等效)

    Returns
    -------
    tonic : array (n,)
    """
    tonic = np.zeros(brain.n, dtype=np.float32)

    # Medulla 細胞 — T4/T5 的直接上游
    medulla_types = ['Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm9']
    n_medulla = 0
    for cell_type in medulla_types:
        cells = t4_t5_result['upstream'].get(cell_type, np.array([], dtype=np.int32))
        if len(cells) > 0:
            tonic[cells] = medulla_tonic
            n_medulla += len(cells)

    # 也給 L1-L5 lamina 一點 tonic(它們已有 lamina_bias=12,但只給 L1/L2/L3/L5)
    # L4 沒在 lamina list 裡(prepare.py 只列 L1/L2/L3/L5)
    l4 = t4_t5_result['upstream'].get('L4', np.array([], dtype=np.int32))
    if len(l4) > 0:
        tonic[l4] = medulla_tonic * 0.5  # 較小的 tonic

    return tonic, n_medulla


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
    """跑單一方向的 grating"""
    seq, times = stimulus.generate_direction(direction_deg, duration_ms)
    dt = stimulus.DT
    batch_steps = int(round(batch_ms / dt))

    n_t4_t5 = len(t4_t5_indices)
    total_spikes = np.zeros(n_t4_t5, dtype=np.int32)

    n_batches = len(seq) // batch_steps
    t_start = time.time()

    for b in range(n_batches):
        batch = seq[b * batch_steps : (b + 1) * batch_steps]
        mean_lum = batch.mean(axis=0)
        counts, _ = brain.step(mean_lum, batch_ms, learning=False)
        total_spikes += counts[t4_t5_indices]

        if b % 20 == 0:
            elapsed = time.time() - t_start
            progress = (b + 1) / n_batches
            print(f"    {progress*100:.1f}% ({(b+1)*batch_ms:.0f}ms, {elapsed:.1f}s)",
                  flush=True)

    return total_spikes


def test_tonic_levels(brain, stimulus, t4_t5_result, tonic_levels):
    """測試不同 tonic 強度,找最小的能讓 T4/T5 spike 的值"""
    print("=" * 70)
    print("Tonic Drive 校準測試")
    print("=" * 70)

    for tonic_val in tonic_levels:
        print(f"\n--- tonic = {tonic_val} ---")
        tonic, n_medulla = build_tonic(brain, t4_t5_result, medulla_tonic=tonic_val)

        # 重建 brain with tonic
        from neural.brain import MemoryBrain
        brain_test = MemoryBrain(GRAPH, eta=0.0, tonic=tonic)

        # 跑 500ms 0° grating
        brain_test.reset(keep_memory=False)
        seq, _ = stimulus.generate_direction(0, duration_ms=500)

        total_t4_t5 = np.zeros(len(t4_t5_result["t4_t5"]), dtype=np.int32)
        for batch_start in range(0, 5000, 100):
            batch = seq[batch_start:batch_start+1000]
            mean_lum = batch.mean(axis=0)
            counts, _ = brain_test.step(mean_lum, 100, learning=False)
            total_t4_t5 += counts[t4_t5_result["t4_t5"]]

        t4_spikes = total_t4_t5[:len(t4_t5_result["t4"])].sum()
        t5_spikes = total_t4_t5[len(t4_t5_result["t4"]):].sum()
        t4_v = brain_test.v[t4_t5_result["t4"]].mean()
        t5_v = brain_test.v[t4_t5_result["t5"]].mean()
        mi1_v = brain_test.v[t4_t5_result["upstream"]["Mi1"]].mean()
        tm3_v = brain_test.v[t4_t5_result["upstream"]["Tm3"]].mean()

        print(f"  Mi1 v: {mi1_v:.2f} mV")
        print(f"  Tm3 v: {tm3_v:.2f} mV")
        print(f"  T4 v: {t4_v:.2f} mV, spikes: {t4_spikes}")
        print(f"  T5 v: {t5_v:.2f} mV, spikes: {t5_spikes}")

        if t4_spikes > 0 or t5_spikes > 0:
            print(f"  ✅ T4/T5 有 spike!")
            return tonic_val, brain_test

    return None, None


def main():
    print("=" * 70)
    print("MaleCNS Drifting Grating 實驗 v3 — Medulla Tonic Drive")
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

    # 校準 tonic 強度
    print("\n--- 校準:測試不同 tonic 強度 ---")
    tonic_levels = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    best_tonic, brain_test = test_tonic_levels(
        brain, stimulus, t4_t5_result, tonic_levels
    )

    if best_tonic is None:
        print("\n❌ 即使 tonic=7,T4/T5 仍無 spike")
        print("   需要考慮方向 B(降低閾值)或方向 C(直接刺激上游)")
        return

    print(f"\n✅ 選擇 tonic = {best_tonic}")

    # 用最佳 tonic 跑完整 8 方向實驗
    tonic, n_medulla = build_tonic(brain, t4_t5_result, medulla_tonic=best_tonic)
    brain_final = MemoryBrain(GRAPH, eta=0.0, tonic=tonic)
    print(f"\n--- 用 tonic={best_tonic} 跑完整實驗 ---")
    print(f"  Medulla 細胞數: {n_medulla}")

    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    duration_ms = 2000
    n_t4_t5 = len(t4_t5)
    n_dirs = len(directions)
    spike_counts = np.zeros((n_t4_t5, n_dirs), dtype=np.int32)

    for d_idx, direction in enumerate(directions):
        print(f"\n  方向 {direction}° ({d_idx+1}/{n_dirs}):")
        brain_final.reset(keep_memory=False)
        counts = run_direction(brain_final, stimulus, direction, t4_t5,
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
    else:
        print("❌ NO-GO: T4/T5 仍無 spike")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / "day6_v3_dsi_results.npz",
        t4_dsi=t4_dsi, t5_dsi=t5_dsi,
        t4_rates=t4_rates, t5_rates=t5_rates,
        t4_pref_dir=pref_dir[:n_t4], t5_pref_dir=pref_dir[n_t4:],
        directions=directions, spike_counts=spike_counts,
        tonic=best_tonic,
    )
    report = {
        "experiment": "MaleCNS frozen drifting grating v3 (medulla tonic)",
        "n_t4": n_t4, "n_t5": n_t5,
        "directions": directions,
        "duration_per_direction_ms": duration_ms,
        "medulla_tonic": best_tonic,
        "t4_dsi_median": float(np.median(t4_dsi)),
        "t4_dsi_mean": float(t4_dsi.mean()),
        "t4_dsi_gt_0_3_pct": float((t4_dsi > 0.3).mean() * 100),
        "t4_dsi_gt_0_6_pct": float((t4_dsi > 0.6).mean() * 100),
        "t4_mean_rate_hz": float(t4_rates.mean()),
        "t4_max_rate_hz": float(t4_rates.max()),
        "t5_dsi_median": float(np.median(t5_dsi)),
        "t5_dsi_mean": float(t5_dsi.mean()),
        "t5_dsi_gt_0_3_pct": float((t5_dsi > 0.3).mean() * 100),
        "t5_dsi_gt_0_6_pct": float((t5_dsi > 0.6).mean() * 100),
        "t5_mean_rate_hz": float(t5_rates.mean()),
        "t5_max_rate_hz": float(t5_rates.max()),
        "biological_reference": "Maisak et al. 2013, Nature, DSI 0.6-0.8",
        "t4_in_biological_range": bool(t4_in_range),
        "t5_in_biological_range": bool(t5_in_range),
    }
    (output_path / "day6_v3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n💾 結果已儲存至 {output_path}/day6_v3_*")


if __name__ == "__main__":
    main()
