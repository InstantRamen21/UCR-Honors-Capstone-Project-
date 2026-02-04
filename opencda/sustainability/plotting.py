# plotting.py
# Place at: C:\Users\goton\OpenCDA\opencda\sustainability\plotting.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_vehicle_energy(csv_path):
    if not os.path.exists(csv_path):
        print("Missing:", csv_path); return
    df = pd.read_csv(csv_path)
    if 'energy_j' in df.columns:
        df['energy_Wh'] = df['energy_j'] / 3600.0
    else:
        # if cumulative was logged instead
        df['energy_Wh'] = df['cumulative_energy_j'] / 3600.0
    plt.figure(figsize=(8,4))
    plt.plot(df['timestamp'], df['energy_Wh'])
    plt.xlabel('time (s)'); plt.ylabel('Energy (Wh)')
    plt.title(os.path.basename(csv_path))
    out = csv_path.replace('.csv','_energy.png')
    plt.savefig(out); plt.close()
    print("Saved:", out)

def plot_eco_scores(summary_json):
    if not os.path.exists(summary_json):
        print("Missing:", summary_json); return
    s = json.load(open(summary_json))
    vehicles = s.get('vehicles', [])
    ids = [v['vehicle_id'] for v in vehicles]
    scores = [v.get('eco_score', 0) for v in vehicles]
    plt.figure(figsize=(8,4))
    plt.bar([str(i) for i in ids], scores)
    plt.xlabel('vehicle id'); plt.ylabel('eco score'); plt.title('Eco Score per vehicle')
    out = os.path.join(os.path.dirname(summary_json), 'eco_scores.png')
    plt.savefig(out); plt.close()
    print("Saved:", out)

def plot_grid_heatmap(summary_json, cell_size=50):
    if not os.path.exists(summary_json):
        print("Missing:", summary_json); return
    s = json.load(open(summary_json))
    grid = s.get('grid', [])
    if not grid:
        print("No grid data"); return
    xs = [c['i'] for c in grid]; ys = [c['j'] for c in grid]; vals = [c['co2_g'] for c in grid]
    xi = np.array(xs); yi = np.array(ys); zi = np.array(vals)
    # build matrix
    minx, maxx = xi.min(), xi.max()
    miny, maxy = yi.min(), yi.max()
    w = maxx - minx + 1; h = maxy - miny + 1
    mat = np.zeros((h, w))
    for x,y,v in zip(xi, yi, zi):
        mat[y-miny, x-minx] = v
    plt.figure(figsize=(6,6))
    plt.imshow(mat, origin='lower')
    plt.colorbar(label='co2_g (proxy)')
    out = os.path.join(os.path.dirname(summary_json), 'grid_heatmap.png')
    plt.savefig(out); plt.close()
    print("Saved:", out)

def plot_eco_scores_timeseries(log_dir):
    csvs = [f for f in os.listdir(log_dir) if f.endswith("_sustain.csv")]
    if not csvs:
        print("No sustainability CSVs found")
        return

    plt.figure(figsize=(9,5))

    for f in csvs:
        path = os.path.join(log_dir, f)
        df = pd.read_csv(path)
        if 'eco_score' not in df.columns:
            continue
        label = f.replace("vehicle_", "").replace("_sustain.csv", "")
        plt.plot(df['timestamp'], df['eco_score'], label=f"veh {label}")

    plt.xlabel("Time (s)")
    plt.ylabel("Eco score")
    plt.title("Eco-driving score over time")
    plt.legend()
    plt.grid(True)

    out = os.path.join(log_dir, "eco_scores.png")
    plt.savefig(out)
    plt.close()
    print("Saved:", out)
