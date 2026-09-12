"""方向 C:直接刺激 Mi1,跳過 lamina

核心問題:MaleCNS 連接體的 Mi1→T4 迴路,在有適當時間延遲刺激時,
能否產生方向選擇性?

HR (Hassenstein-Reichardt) 模型:
- 給 Mi1 一個移動的亮度波
- 不同位置的 Mi1 在不同時間被激活
- T4 收到來自相鄰位置 Mi1 的延遲輸入
- preferred direction:延遲匹配 → T4 強響應
- null direction:延遲不匹配 → T4 弱響應

實作:
- 直接給 Mi1 一個 tonic drive(取代光感受器→lamina→Mi1 路徑)
- Mi1 的 drive 強度 = f(空間位置, 時間, 方向)
- 移動方向 = 波的傳播方向
- 用 brain.step(stimulation=[(mi1_indices, drive_values)])
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neural.common import DATA, GRAPH
from neural.t4_t5 import identify_t4_t5
from neural.transmitters import transmitter_signs
import pyarrow.feather as feather
import pandas as pd


def get_mi1_spatial_layout(ids, t4_t5_result):
    """取得 Mi1 的空間座標

    Returns
    -------
    mi1_indices : array of valid Mi1 node indices
    mi1_xy : array (N, 2) cartesian coords
    """
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
                          contrast=1.0, mean_drive=5.0):
    """生成移動亮度波,直接驅動 Mi1

    Parameters
    ----------
    mi1_indices : array
        Mi1 node indices
    mi1_xy : array (N, 2)
        Mi1 cartesian coords
    direction_deg : float
        波的移動方向
    t_ms : float
        當前時間(ms)
    spatial_freq : float
        空間頻率(cycle per hex unit)
    temporal_freq : float
        時間頻率(Hz)
    contrast : float
        對比度(0-1)
    mean_drive : float
        平均驅動強度

    Returns
    -------
    drive : array (N,)
        每個 Mi1 的驅動強度
    """
    direction_rad = np.deg2rad(direction_deg)
    dx = np.cos(direction_rad)
    dy = np.sin(direction_rad)
    # 每個 Mi1 在方向軸上的投影位置
    projection = mi1_xy[:, 0] * dx + mi1_xy[:, 1] * dy
    # 時間相位
    phase = 2 * np.pi * temporal_freq * (t_ms / 1000.0)
    # 亮度波: mean + contrast * sin(2π * sf * projection - phase)
    # 減號讓波沿方向軸移動(preferred direction)
    wave = mean_drive + contrast * mean_drive * np.sin(
        2 * np.pi * spatial_freq * projection - phase
    )
    return np.clip(wave, 0, None).astype(np.float32)


def run_direction_hr(brain, mi1_indices, mi1_xy, direction_deg,
                      duration_ms=1000, batch_ms=10,
                      mean_drive=5.0, t4_t5_result=None):
    """跑單一方向的移動波刺激

    直接用 stimulation 參數驅動 Mi1,不用 luminance 路徑
    """
    t4 = t4_t5_result["t4"]
    t5 = t4_t5_result["t5"]
    t4_t5 = t4_t5_result["t4_t5"]

    n_batches = int(duration_ms / batch_ms)
    total_t4_spikes = np.zeros(len(t4), dtype=np.int32)
    total_t5_spikes = np.zeros(len(t5), dtype=np.int32)

    # 用一個 dummy luminance(因為 brain.step 需要它)
    dummy_lum = np.zeros(len(brain.retina), dtype=np.float32)

    t_start = time.time()
    for b in range(n_batches):
        t_ms = b * batch_ms
        # 生成這個 batch 的 Mi1 驅動
        mi1_drive = moving_wave_stimulus(
            mi1_indices, mi1_xy, direction_deg, t_ms,
            mean_drive=mean_drive
        )
        # 用 stimulation 參數直接驅動 Mi1
        counts, _ = brain.step(
            dummy_lum, batch_ms, learning=False,
            stimulation=[(mi1_indices, mi1_drive)]
        )
        total_t4_spikes += counts[t4]
        total_t5_spikes += counts[t5]

        if b % 20 == 0:
            elapsed = time.time() - t_start
            print(f"    {b*batch_ms:4d}ms ({(b+1)/n_batches*100:.0f}%, {elapsed:.1f}s)",
                  flush=True)

    return total_t4_spikes, total_t5_spikes


def main():
    print("=" * 70)
    print("方向 C: 直接刺激 Mi1,跳過 lamina")
    print("HR 模型: 移動亮度波 → Mi1 → T4")
    print("=" * 70)

    # 載入
    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    mi1_indices, mi1_xy = get_mi1_spatial_layout(ids, t4_t5_result)
    print(f"\nMi1 有座標: {len(mi1_indices)}")
    print(f"X: [{mi1_xy[:,0].min():.1f}, {mi1_xy[:,0].max():.1f}]")
    print(f"Y: [{mi1_xy[:,1].min():.1f}, {mi1_xy[:,1].max():.1f}]")

    from neural.brain import MemoryBrain

    # 用較小的 tonic,讓 Mi1 的驅動主要來自我們的 stimulation
    # 只給 Mi1 一個基線驅動,讓它接近閾值但不持續激發
    tonic = np.zeros(166700, dtype=np.float32)
    # 不給 Mi1 tonic — 完全由 stimulation 驅動
    # 但給 Mi4 一點 tonic(它是抑制性,幫助 T4 達到平衡)
    tonic[t4_t5_result["upstream"]["Mi4"]] = 3.0

    brain = MemoryBrain(GRAPH, eta=0.0, tonic=tonic)

    # 先測試 200ms 確認 Mi1 被驅動
    print("\n--- Pipeline 測試: 200ms 0° 移動波 ---")
    brain.reset(keep_memory=False)
    t4_s, t5_s = run_direction_hr(
        brain, mi1_indices, mi1_xy, 0,
        duration_ms=200, mean_drive=8.0, t4_t5_result=t4_t5_result
    )
    print(f"  T4 spikes: {t4_s.sum()}, T5 spikes: {t5_s.sum()}")
    print(f"  T4 有 spike 的細胞: {np.count_nonzero(t4_s)}")
    print(f"  T5 有 spike 的細胞: {np.count_nonzero(t5_s)}")

    if t4_s.sum() == 0 and t5_s.sum() == 0:
        print("\n  Mi1 沒被驅動到 spike,增加 mean_drive")
        # 測試更大 drive
        for drive in [10, 12, 15, 20]:
            brain.reset(keep_memory=False)
            t4_s, t5_s = run_direction_hr(
                brain, mi1_indices, mi1_xy, 0,
                duration_ms=200, mean_drive=drive, t4_t5_result=t4_t5_result
            )
            mi1_spikes = 0
            # 簡單檢查 Mi1 是否 spike
            print(f"  drive={drive}: T4={t4_s.sum()}, T5={t5_s.sum()}")
            if t4_s.sum() > 0 or t5_s.sum() > 0:
                print(f"  ✅ drive={drive} 有效!")
                best_drive = drive
                break
        else:
            print("  ❌ 即使 drive=20 仍無 spike")
            return
    else:
        best_drive = 8.0

    # 跑 8 方向
    print(f"\n--- 完整 8 方向實驗 (drive={best_drive}, 500ms) ---")
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    duration_ms = 500

    t4_rates_all = []
    t5_rates_all = []

    for direction in directions:
        print(f"\n  方向 {direction}°:")
        brain.reset(keep_memory=False)
        t4_s, t5_s = run_direction_hr(
            brain, mi1_indices, mi1_xy, direction,
            duration_ms=duration_ms, mean_drive=best_drive,
            t4_t5_result=t4_t5_result
        )
        t4_rate = t4_s / (duration_ms / 1000)  # Hz per cell
        t5_rate = t5_s / (duration_ms / 1000)
        t4_rates_all.append(t4_rate)
        t5_rates_all.append(t5_rate)
        print(f"    T4 avg={t4_rate.mean():.2f} Hz, max={t4_rate.max():.0f} Hz, "
              f"active={np.count_nonzero(t4_s)}/{len(t4_s)}")
        print(f"    T5 avg={t5_rate.mean():.2f} Hz, max={t5_rate.max():.0f} Hz, "
              f"active={np.count_nonzero(t5_s)}/{len(t5_s)}")

    t4_rates_all = np.array(t4_rates_all)  # (8, n_t4)
    t5_rates_all = np.array(t5_rates_all)  # (8, n_t5)

    # 計算 DSI
    def compute_dsi(rates_matrix, directions):
        n_dirs, n_cells = rates_matrix.shape
        dsi = np.zeros(n_cells, dtype=np.float32)
        pref_dir = np.zeros(n_cells, dtype=np.int32)
        for i in range(n_cells):
            rates = rates_matrix[:, i]
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

    # 報告
    print("\n" + "=" * 70)
    print("方向 C DSI 結果")
    print("=" * 70)

    # 只對有 spike 的細胞算 DSI
    t4_active = np.array([t4_rates_all[:, i].sum() > 0 for i in range(len(t4_dsi))])
    t5_active = np.array([t5_rates_all[:, i].sum() > 0 for i in range(len(t5_dsi))])

    print(f"\n--- T4 全部 (n={len(t4_dsi)}) ---")
    print(f"  有 spike 的細胞: {t4_active.sum()}")
    print(f"  DSI 中位數 (全部): {np.median(t4_dsi):.3f}")
    if t4_active.sum() > 0:
        print(f"  DSI 中位數 (有 spike): {np.median(t4_dsi[t4_active]):.3f}")
        print(f"  DSI > 0.3: {(t4_dsi[t4_active] > 0.3).mean()*100:.1f}%")
        print(f"  DSI > 0.6: {(t4_dsi[t4_active] > 0.6).mean()*100:.1f}%")

    print(f"\n--- T5 全部 (n={len(t5_dsi)}) ---")
    print(f"  有 spike 的細胞: {t5_active.sum()}")
    print(f"  DSI 中位數 (全部): {np.median(t5_dsi):.3f}")
    if t5_active.sum() > 0:
        print(f"  DSI 中位數 (有 spike): {np.median(t5_dsi[t5_active]):.3f}")
        print(f"  DSI > 0.3: {(t5_dsi[t5_active] > 0.3).mean()*100:.1f}%")
        print(f"  DSI > 0.6: {(t5_dsi[t5_active] > 0.6).mean()*100:.1f}%")

    # 各方向平均激發率
    print(f"\n--- 各方向平均激發率 ---")
    print(f"  {'方向':>4s}  {'T4 avg':>8s}  {'T5 avg':>8s}")
    for i, d in enumerate(directions):
        print(f"  {d:4d}°  {t4_rates_all[i].mean():8.2f}  {t5_rates_all[i].mean():8.2f}")

    # 偏好方向分布
    if t4_active.sum() > 0:
        print(f"\n--- T4 偏好方向分布 (有 spike 的) ---")
        for d in directions:
            count = ((t4_pref == d) & t4_active).sum()
            print(f"  {d:3d}°: {count:5d} cells")

    # 對照 Maisak 2013
    print(f"\n--- 對照 Maisak 2013 (Nature) ---")
    print(f"  真實 T4/T5 DSI: 0.6-0.8 (GCaMP 電生理)")
    if t4_active.sum() > 0:
        print(f"  模擬 T4 DSI 中位數 (有 spike): {np.median(t4_dsi[t4_active]):.3f}")
    if t5_active.sum() > 0:
        print(f"  模擬 T5 DSI 中位數 (有 spike): {np.median(t5_dsi[t5_active]):.3f}")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    np.savez(
        output_path / "day6_directionC_dsi.npz",
        t4_dsi=t4_dsi, t5_dsi=t5_dsi,
        t4_rates=t4_rates_all.T, t5_rates=t5_rates_all.T,
        t4_pref=t4_pref, t5_pref=t5_pref,
        t4_active=t4_active, t5_active=t5_active,
        directions=directions,
        mean_drive=best_drive,
    )
    print(f"\n💾 結果已儲存至 {output_path}/day6_directionC_dsi.npz")


if __name__ == "__main__":
    main()
