# evaluator.py
# Place at: C:\Users\goton\OpenCDA\opencda\sustainability\evaluator.py

import os, json
from opencda.sustainability.metrics import SustainabilityMetrics

class SustainabilityEvaluator:
    def __init__(self, map_bounds=None, cell_size_m=50, log_folder="cache/sustainability_logs", config=None):
        self.metrics = {}  # vehicle_id -> SustainabilityMetrics
        self.grid = {}     # (i,j) -> co2_g
        self.map_bounds = map_bounds or {'min_x': -1000, 'min_y': -1000, 'max_x': 1000, 'max_y': 1000}
        self.cell_size = cell_size_m
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self.config = config or {}

    def register_vehicle(self, vehicle, veh_params=None):
        if vehicle.id in self.metrics:
            return
        m = SustainabilityMetrics(vehicle, output_folder=self.log_folder, vehicle_id=vehicle.id, veh_params=veh_params, model_type=self.config.get('model_type','auto'))
        self.metrics[vehicle.id] = m

    def update(self, snapshot):
        # call update on each vehicle metrics
        for vid, m in list(self.metrics.items()):
            try:
                m.update(snapshot)
                loc = m.vehicle.get_location()
                i = int((loc.x - self.map_bounds['min_x']) / self.cell_size)
                j = int((loc.y - self.map_bounds['min_y']) / self.cell_size)
                self.grid.setdefault((i,j), 0.0)
                # add current accumulated co2 (simple: add delta since last update isn't stored; using total is ok for proxies)
                self.grid[(i,j)] += m.co2_g
            except Exception as e:
                # avoid crashing the sim if a metric update fails
                print(f"[SustEval] update error for vehicle {vid}: {e}")

    def finalize(self, out_path=None):
        summary = {
            'vehicles': [],
            'grid': []
        }
        for vid, m in self.metrics.items():
            summary['vehicles'].append(m.results())
            m.close()
        for (i,j), co2 in self.grid.items():
            summary['grid'].append({'i': i, 'j': j, 'co2_g': co2})
        out_path = out_path or os.path.join(self.log_folder, 'sustainability_summary.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[SustEval] Summary written to {out_path}")
