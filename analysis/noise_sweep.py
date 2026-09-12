"""Noise sweep:三組(MaleCNS / Config / ER)在不同 noise level 下的 DSI 變化

關鍵問題:加 noise 後,MaleCNS 是否仍然顯著高於 Config/ER?
         MaleCNS 的 DSI 是否會降到 0.6-0.8 生物學區間?

公平性:三組用完全相同的 noise level。
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


def moving_wave_noisy(mi1_xy, direction_deg, t_ms, rng,
                       sf=0.05, tf=1.0, drive=15.0, noise_std=0.0):
    """移動波 + Gaussian noise"""
    rad = np.deg2rad(direction_deg)
    proj = mi1_xy[:, 0] * np.cos(rad) + mi1_xy[:, 1] * np.sin(rad)
    phase = 2 * np.pi * tf * (t_ms / 1000.0)
    wave = drive + drive * np.sin(2 * np.pi * sf * proj - phase)
    if noise_std > 0:
        wave = wave + rng.normal(0, noise_std * drive, size=len(wave))
    return np.clip(wave, 0, None).astype(np.float32)


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
              duration_ms=300, drive=15.0, sf=0.05, tf=1.0, noise_std=0.0,
              noise_seed=0):
    t4 = t4_t5_result["t4"]
    tonic = np.zeros(brain.n, dtype=np.float32)
    tonic[t4_t5_result["upstream"]["Mi4"]] = 3.0
    brain.tonic = tonic
    dummy_lum = np.zeros(len(brain.retina), dtype=np.float32)
    rng = np.random.RandomState(noise_seed)
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    n_batches = int(duration_ms / 10)
    rates_all = []
    for direction in directions:
        brain.reset(keep_memory=False)
        counts = np.zeros(len(t4), dtype=np.int32)
        for b in range(n_batches):
            mi1_drive = moving_wave_noisy(
                mi1_xy, direction, b * 10, rng,
                sf=sf, tf=tf, drive=drive, noise_std=noise_std
            )
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
        "dsi_gt_06_pct": float((dsi_a > 0.6).mean() * 100) if len(dsi_a) > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Noise Sweep: 三組在不同 noise level 下的 DSI")
    print("=" * 70)

    ids = np.load(DATA / "normalized" / "neuron_ids.npy")
    t4_t5_result = identify_t4_t5(ids)
    mi1_indices, mi1_xy = get_mi1_spatial_layout(ids, t4_t5_result)

    params = {"drive": 15.0, "sf": 0.05, "tf": 1.0, "duration_ms": 300}
    noise_levels = [0.0, 0.1, 0.2, 0.5]
    n_seeds = 3  # Config/ER 用 3 seeds(時間考量)

    all_results = {}

    for noise_std in noise_levels:
        print(f"\n{'='*60}")
        print(f"Noise level = {noise_std}")
        print(f"{'='*60}")

        # MaleCNS 原版
        brain = MemoryBrain(GRAPH, eta=0.0)
        t0 = time.time()
        dsi, pref, active, rates = run_8dir(
            brain, mi1_indices, mi1_xy, t4_t5_result,
            noise_std=noise_std, noise_seed=0, **params
        )
        s_orig = summarize(dsi, active)
        print(f"MaleCNS: active={s_orig['n_active']}, DSI={s_orig['dsi_median']:.3f} ({time.time()-t0:.0f}s)")

        # Config (3 seeds)
        config_summaries = []
        for seed in range(n_seeds):
            brain = MemoryBrain(GRAPH, eta=0.0)
            apply_config_model(brain, seed)
            t0 = time.time()
            dsi, pref, active, rates = run_8dir(
                brain, mi1_indices, mi1_xy, t4_t5_result,
                noise_std=noise_std, noise_seed=seed, **params
            )
            s = summarize(dsi, active)
            config_summaries.append(s)

        c_dsi = [s['dsi_median'] for s in config_summaries]
        c_act = [s['n_active'] for s in config_summaries]
        print(f"Config: active={np.mean(c_act):.0f}±{np.std(c_act):.0f}, "
              f"DSI={np.mean(c_dsi):.3f}±{np.std(c_dsi):.3f}")

        # ER (3 seeds)
        er_summaries = []
        for seed in range(n_seeds):
            brain = MemoryBrain(GRAPH, eta=0.0)
            apply_er_model(brain, seed)
            t0 = time.time()
            dsi, pref, active, rates = run_8dir(
                brain, mi1_indices, mi1_xy, t4_t5_result,
                noise_std=noise_std, noise_seed=seed, **params
            )
            s = summarize(dsi, active)
            er_summaries.append(s)

        e_dsi = [s['dsi_median'] for s in er_summaries]
        e_act = [s['n_active'] for s in er_summaries]
        print(f"ER:     active={np.mean(e_act):.0f}±{np.std(e_act):.0f}, "
              f"DSI={np.mean(e_dsi):.3f}±{np.std(e_dsi):.3f}")

        all_results[noise_std] = {
            "malecns": s_orig,
            "config": {"dsi_mean": float(np.mean(c_dsi)), "dsi_std": float(np.std(c_dsi)),
                       "active_mean": float(np.mean(c_act))},
            "er": {"dsi_mean": float(np.mean(e_dsi)), "dsi_std": float(np.std(e_dsi)),
                   "active_mean": float(np.mean(e_act))},
        }

    # 彙總表
    print(f"\n{'='*70}")
    print("Noise Sweep 彙總")
    print(f"{'='*70}")
    print(f"\n{'noise':>6s} {'MaleCNS':>12s} {'Config':>16s} {'ER':>16s} {'M>C?':>6s} {'C>E?':>6s}")
    print("-" * 70)
    for noise in noise_levels:
        r = all_results[noise]
        m = r['malecns']['dsi_median']
        c = r['config']['dsi_mean']
        c_s = r['config']['dsi_std']
        e = r['er']['dsi_mean']
        e_s = r['er']['dsi_std']
        m_gt_c = "✅" if m > c + 2*c_s else "❌"
        c_gt_e = "✅" if c > e + 2*e_s else "❌"
        print(f"{noise:6.2f} {m:12.3f} {c:10.3f}±{c_s:.3f} {e:10.3f}±{e_s:.3f} {m_gt_c:>6s} {c_gt_e:>6s}")

    # 找讓 MaleCNS 落在 0.6-0.8 的 noise level
    print(f"\n--- 找讓 MaleCNS DSI 落在 0.6-0.8 的 noise level ---")
    for noise in noise_levels:
        m = all_results[noise]['malecns']['dsi_median']
        if 0.6 <= m <= 0.8:
            print(f"✅ noise={noise}: MaleCNS DSI={m:.3f} (落在生物學區間)")
            c = all_results[noise]['config']['dsi_mean']
            e = all_results[noise]['er']['dsi_mean']
            print(f"   Config={c:.3f}, ER={e:.3f}")
            break
    else:
        print("沒有 noise level 讓 MaleCNS 落在 0.6-0.8")
        print("MaleCNS DSI 在所有 noise level:")
        for noise in noise_levels:
            m = all_results[noise]['malecns']['dsi_median']
            print(f"  noise={noise}: {m:.3f}")

    # 儲存
    output_path = Path("/home/z/my-project/malecns-topology-prior/results")
    (output_path / "noise_sweep_report.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n💾 儲存至 {output_path}/noise_sweep_report.json")


if __name__ == "__main__":
    main()
