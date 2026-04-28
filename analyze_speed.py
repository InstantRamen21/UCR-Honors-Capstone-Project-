# -*- coding: utf-8 -*-
"""
Speed Comparison Analysis
Compares vehicle 45 speed profile between baseline PID and ECO controller runs.
Also computes a simple efficiency score based on speed smoothness.

Usage:
    python analyze_speed.py

Expects these files to exist:
    ./cache/speed_logs/pid_vehicle_49.json
    ./cache/speed_logs/eco_vehicle_49.json
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_log(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    steps        = [d['step']         for d in data]
    speeds       = [d['speed']        for d in data]
    target_speeds = [d['target_speed'] for d in data]
    return np.array(steps), np.array(speeds), np.array(target_speeds)


def compute_efficiency_score(speeds, target_speeds):
    """
    Efficiency score based on speed smoothness and target tracking.
    
    Components:
    - Smoothness (60%): penalises large step-to-step speed changes (jerk proxy)
    - Tracking  (40%): penalises deviation from target speed

    Returns a score 0-100 where higher is better.
    """
    # Smoothness: mean absolute difference between consecutive speeds
    diffs = np.abs(np.diff(speeds))
    mean_jerk = np.mean(diffs)
    # Normalise: 0 jerk -> 100, anything >= 5 km/h/step -> 0
    smoothness = max(0.0, 1.0 - mean_jerk / 5.0) * 100

    # Tracking: mean absolute error from target speed
    mae = np.mean(np.abs(speeds - target_speeds))
    # Normalise: 0 error -> 100, anything >= 10 km/h error -> 0
    tracking = max(0.0, 1.0 - mae / 10.0) * 100

    score = 0.6 * smoothness + 0.4 * tracking
    return round(score, 2), round(smoothness, 2), round(tracking, 2)


def load_eco_score(label):
    """Try to pull eco_score from sustainability_logs for context."""
    path = './cache/sustainability_logs/sustainability_summary.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    # summary may be a list or dict depending on OpenCDA version
    if isinstance(data, list):
        for entry in data:
            if entry.get('vehicle_id') == 45:
                return entry.get('eco_score')
    elif isinstance(data, dict):
        return data.get('eco_score')
    return None


def main():
    baseline_path = './cache/speed_logs/pid_vehicle_49.json'
    eco_path      = './cache/speed_logs/eco_vehicle_49.json'    

    missing = [p for p in [baseline_path, eco_path] if not os.path.exists(p)]
    if missing:
        print("Missing log files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun the simulation twice:")
        print("  1. With standard PID  -> saves vehicle_45_baseline.json")
        print("  2. With ECO controller -> saves vehicle_45_eco.json")
        return

    steps_b, speeds_b, targets_b = load_log(baseline_path)
    steps_e, speeds_e, targets_e = load_log(eco_path)

    score_b, smooth_b, track_b = compute_efficiency_score(speeds_b, targets_b)
    score_e, smooth_e, track_e = compute_efficiency_score(speeds_e, targets_e)

    print("=" * 55)
    print(f"{'Metric':<30} {'Baseline':>10} {'ECO':>10}")
    print("=" * 55)
    print(f"{'Efficiency Score (0-100)':<30} {score_b:>10} {score_e:>10}")
    print(f"{'  Smoothness component':<30} {smooth_b:>10} {smooth_e:>10}")
    print(f"{'  Tracking component':<30} {track_b:>10} {track_e:>10}")
    print(f"{'Mean speed (km/h)':<30} {np.mean(speeds_b):>10.2f} {np.mean(speeds_e):>10.2f}")
    print(f"{'Max speed (km/h)':<30} {np.max(speeds_b):>10.2f} {np.max(speeds_e):>10.2f}")
    print(f"{'Speed std dev':<30} {np.std(speeds_b):>10.2f} {np.std(speeds_e):>10.2f}")
    print(f"{'Steps logged':<30} {len(steps_b):>10} {len(steps_e):>10}")
    print("=" * 55)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Vehicle 45 — Baseline PID vs ECO Controller', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Plot 1: Speed profiles ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps_b, speeds_b,  color='steelblue',  linewidth=1.2, label='Baseline PID speed')
    ax1.plot(steps_e, speeds_e,  color='darkorange',  linewidth=1.2, label='ECO speed')
    ax1.plot(steps_b, targets_b, color='steelblue',  linewidth=0.8,
             linestyle='--', alpha=0.5, label='Baseline target')
    ax1.plot(steps_e, targets_e, color='darkorange',  linewidth=0.8,
             linestyle='--', alpha=0.5, label='ECO target')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Speed Profile Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Step-to-step speed change (jerk proxy) ---
    ax2 = fig.add_subplot(gs[1, :])
    jerk_b = np.abs(np.diff(speeds_b))
    jerk_e = np.abs(np.diff(speeds_e))
    ax2.plot(steps_b[1:], jerk_b, color='steelblue',  linewidth=0.9, alpha=0.8, label='Baseline')
    ax2.plot(steps_e[1:], jerk_e, color='darkorange',  linewidth=0.9, alpha=0.8, label='ECO')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('|Δspeed| (km/h)')
    ax2.set_title('Speed Change Per Step (lower = smoother)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Efficiency score bar chart ---
    ax3 = fig.add_subplot(gs[2, 0])
    bars = ax3.bar(['Baseline', 'ECO'], [score_b, score_e],
                   color=['steelblue', 'darkorange'], width=0.4)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('Score (0-100)')
    ax3.set_title('Speed Efficiency Score')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [score_b, score_e]):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, str(val),
                 ha='center', va='bottom', fontweight='bold')

    # --- Plot 4: Speed distribution histogram ---
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(speeds_b, bins=30, color='steelblue',  alpha=0.6, label='Baseline', density=True)
    ax4.hist(speeds_e, bins=30, color='darkorange',  alpha=0.6, label='ECO',      density=True)
    ax4.set_xlabel('Speed (km/h)')
    ax4.set_ylabel('Density')
    ax4.set_title('Speed Distribution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig('./cache/speed_logs/speed_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to ./cache/speed_logs/speed_comparison.png")
    plt.show()


if __name__ == '__main__':
    main()