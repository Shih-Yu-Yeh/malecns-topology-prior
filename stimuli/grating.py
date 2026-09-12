"""Hex grid drifting grating 生成器

在果蠅複眼的 hex 座標系上直接生成 drifting grating,避免矩形取樣偏差。

背景:
- 果蠅複眼由約 750-800 個小眼組成,以六角形密排排列
- R1-R6 光感受器因 neural superposition 投射到同一 lamina cartridge
- MaleCNS annotations 中的 assignedOlHex1/Hex2 是 axial hex 座標
- Stonkfly 的 prepare.py 把 hex 轉成 cartesian: x = h0 - 0.5*h1, y = sqrt(3)/2*h1

實作:
- 直接在 hex cartesian 座標(x, y)上生成正弦光柵
- grating 方向由 angle 參數控制
- 每個時間步,grating 沿方向軸移動

參考:A Connectome Based Hexagonal Lattice Convolutional Network Model
of the Drosophila Visual System (arXiv:1806.04793)
"""

import numpy as np
from pathlib import Path


def hex_to_cartesian(hexes):
    """Axial hex 座標轉 cartesian

    Parameters
    ----------
    hexes : array of shape (N, 2)
        axial hex coordinates (q, r)

    Returns
    -------
    xy : array of shape (N, 2)
        cartesian (x, y) in hex grid units
    """
    q = hexes[:, 0].astype(np.float64)
    r = hexes[:, 1].astype(np.float64)
    x = q - 0.5 * r
    y = np.sqrt(3) / 2 * r
    return np.column_stack([x, y])


def drifting_grating(retina_xy, direction_deg, spatial_freq, temporal_freq,
                     phase=0.0, contrast=1.0, mean_lum=0.5):
    """在 hex cartesian 座標上生成 drifting grating 亮度值

    Parameters
    ----------
    retina_xy : array of shape (N, 2)
        每個 R1-R6 光感受器的 cartesian 座標
    direction_deg : float
        grating 移動方向(度數,0=右移,90=上移)
    spatial_freq : float
        空間頻率(cycle per hex grid unit)
    temporal_freq : float
        時間頻率(Hz) — grating 移動速度
    phase : float
        初始相位(radians)
    contrast : float
        對比度(0-1)
    mean_lum : float
        平均亮度(0-1)

    Returns
    -------
    luminance : array of shape (N,)
        每個光感受器在此時刻的亮度值(0-1)
    """
    direction_rad = np.deg2rad(direction_deg)
    # 方向單位向量
    dx = np.cos(direction_rad)
    dy = np.sin(direction_rad)
    # 每個光感受器在方向軸上的投影位置
    projection = retina_xy[:, 0] * dx + retina_xy[:, 1] * dy
    # grating: L = mean + contrast * sin(2π * sf * projection + phase)
    # 時間項隱含在 phase 裡: phase = 2π * tf * t
    luminance = mean_lum + contrast * mean_lum * np.sin(
        2 * np.pi * spatial_freq * projection + phase
    )
    return np.clip(luminance, 0, 1).astype(np.float32)


def grating_sequence(retina_xy, direction_deg, duration_ms, dt=0.1,
                     spatial_freq=0.05, temporal_freq=2.0, contrast=1.0,
                     mean_lum=0.5, phase_offset=0.0):
    """生成一段時間的 drifting grating 序列

    Parameters
    ----------
    retina_xy : array of shape (N, 2)
        R1-R6 cartesian 座標
    direction_deg : float
        移動方向(度)
    duration_ms : float
        總時長(ms)
    dt : float
        時間步長(ms)
    spatial_freq : float
        空間頻率(cycle per hex unit)
    temporal_freq : float
        時間頻率(Hz)
    contrast : float
        對比度
    mean_lum : float
        平均亮度
    phase_offset : float
        初始相位

    Returns
    -------
    sequence : array of shape (T, N)
        T = duration_ms / dt 個時間步,每步 N 個光感受器亮度
    times : array of shape (T,)
        每步的時間(ms)
    """
    n_steps = int(round(duration_ms / dt))
    times = np.arange(n_steps) * dt
    # 時間相位: 2π * tf * t (t in seconds)
    phases = 2 * np.pi * temporal_freq * (times / 1000.0) + phase_offset

    sequence = np.zeros((n_steps, len(retina_xy)), dtype=np.float32)
    for i, phase in enumerate(phases):
        sequence[i] = drifting_grating(
            retina_xy, direction_deg, spatial_freq, temporal_freq,
            phase=phase, contrast=contrast, mean_lum=mean_lum
        )
    return sequence, times


class GratingStimulus:
    """完整的 drifting grating 實驗刺激

    8 個方向(0, 45, 90, ..., 315),每個方向 2 秒
    符合 Allen Brain Observatory 標準協議
    """

    DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    DURATION_MS = 2000.0  # 每方向 2 秒
    DT = 0.1  # 0.1 ms 步長(與 Stonkfly LIF 核心一致)
    SPATIAL_FREQ = 0.05  # cycle per hex unit
    TEMPORAL_FREQ = 2.0  # Hz
    CONTRAST = 1.0
    MEAN_LUM = 0.5

    def __init__(self, graph_path):
        """載入 graph.npz,取得 retina 座標

        Parameters
        ----------
        graph_path : Path
            graph.npz 路徑
        """
        g = np.load(graph_path)
        self.retina = g["retina"]  # R1-R6 node indices
        self.hexes = g["hexes"]    # axial hex coords (N, 2)
        self.retina_xy = hex_to_cartesian(self.hexes)

        # 統計
        self.n_receptors = len(self.retina)
        self.xy_extent = {
            "x_min": float(self.retina_xy[:, 0].min()),
            "x_max": float(self.retina_xy[:, 0].max()),
            "y_min": float(self.retina_xy[:, 1].min()),
            "y_max": float(self.retina_xy[:, 1].max()),
        }

        # 分左右眼(用 rootSide)
        # Stonkfly 的 prepare.py 用 uv 的 x 座標分左右
        # uv x < 0.4 是右眼,uv x > 0.4 是左眼(參考 prepare.py 第 71 行)
        uv = g["uv"]
        self.left_mask = uv[:, 0] > 0.4
        self.right_mask = uv[:, 0] < 0.4
        # 注意: Stonkfly 用 0.6*z (L) 和 0.4+0.6*(1-z) (R)
        # 所以 L 的 uv.x 在 [0, 0.6], R 的 uv.x 在 [0.4, 1.0]
        # 重疊區 [0.4, 0.6] — 我們簡化用 0.5 分界
        self.left_mask = uv[:, 0] >= 0.5
        self.right_mask = uv[:, 0] < 0.5

    def generate_direction(self, direction_deg, duration_ms=None):
        """生成單一方向的 grating 序列

        Returns
        -------
        sequence : array (T, N) — T 时间步, N 光感受器
        times : array (T,)
        """
        if duration_ms is None:
            duration_ms = self.DURATION_MS
        return grating_sequence(
            self.retina_xy, direction_deg, duration_ms, dt=self.DT,
            spatial_freq=self.SPATIAL_FREQ, temporal_freq=self.TEMPORAL_FREQ,
            contrast=self.CONTRAST, mean_lum=self.MEAN_LUM
        )

    def generate_all(self):
        """生成所有 8 個方向的 grating

        Returns
        -------
        stimuli : dict {direction: (sequence, times)}
        """
        stimuli = {}
        for d in self.DIRECTIONS:
            print(f"  生成 {d}° grating...", flush=True)
            seq, times = self.generate_direction(d)
            stimuli[d] = (seq, times)
            print(f"    形狀: {seq.shape}, 時長: {times[-1]:.0f} ms")
        return stimuli

    def report(self):
        """生成實驗配置報告"""
        return {
            "n_directions": len(self.DIRECTIONS),
            "directions_deg": self.DIRECTIONS,
            "duration_per_direction_ms": self.DURATION_MS,
            "total_duration_ms": self.DURATION_MS * len(self.DIRECTIONS),
            "dt_ms": self.DT,
            "spatial_freq_cycles_per_hex": self.SPATIAL_FREQ,
            "temporal_freq_hz": self.TEMPORAL_FREQ,
            "contrast": self.CONTRAST,
            "mean_luminance": self.MEAN_LUM,
            "n_receptors": self.n_receptors,
            "n_left_eye": int(self.left_mask.sum()),
            "n_right_eye": int(self.right_mask.sum()),
            "xy_extent": self.xy_extent,
            "coordinate_system": "axial hex → cartesian (x=q-0.5r, y=sqrt(3)/2*r)",
            "reference": "Allen Brain Observatory canonical drifting grating paradigm",
            "hex_grid_reference": "arXiv:1806.04793 (Hexagonal Lattice CNN for Drosophila)",
        }


if __name__ == "__main__":
    import json
    import sys
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neural"))
    from common import GRAPH

    print("=" * 70)
    print("Hex Grid Drifting Grating 生成器")
    print("=" * 70)

    stim = GratingStimulus(GRAPH)
    print(f"\n配置報告:")
    print(json.dumps(stim.report(), indent=2))

    # 測試生成一個方向
    print(f"\n--- 測試生成 0° 方向 ---")
    seq, times = stim.generate_direction(0)
    print(f"序列形狀: {seq.shape}")
    print(f"時長: {times[-1]:.1f} ms ({len(times)} steps)")
    print(f"亮度範圍: [{seq.min():.3f}, {seq.max():.3f}]")
    print(f"平均亮度: {seq.mean():.3f}")

    # 驗證方向選擇性:0° 和 180° 應該是反相位
    print(f"\n--- 驗證方向性 ---")
    seq_0, _ = stim.generate_direction(0)
    seq_180, _ = stim.generate_direction(180)
    # 同一時間點,0° 和 180° 的亮度應該是互補的
    mid_t = len(times) // 2
    diff = np.abs(seq_0[mid_t] - seq_180[mid_t])
    print(f"0° vs 180° 在 t={times[mid_t]:.0f}ms 的差異: mean={diff.mean():.3f}, max={diff.max():.3f}")
    # 理論上應該接近 2*contrast*mean_lum = 1.0

    # 生成所有 8 個方向
    print(f"\n--- 生成所有 8 個方向 ---")
    all_stim = stim.generate_all()
    print(f"\n✅ 完成,共 {len(all_stim)} 個方向")
