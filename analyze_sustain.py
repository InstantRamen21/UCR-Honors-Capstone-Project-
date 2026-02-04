# analyze_sustain.py
import glob, os, json
from opencda.sustainability.plotting import plot_vehicle_energy, plot_eco_scores, plot_grid_heatmap

logs = glob.glob("cache/sustainability_logs/vehicle_*_sustain.csv")
for f in logs:
    plot_vehicle_energy(f)

summary = "cache/sustainability_logs/sustainability_summary.json"
if os.path.exists(summary):
    plot_eco_scores(summary)
    plot_grid_heatmap(summary)
else:
    print("Summary not found:", summary)
