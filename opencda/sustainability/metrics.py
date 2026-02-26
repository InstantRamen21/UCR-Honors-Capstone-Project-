# metrics.py
# Improved sustainability metrics for OpenCDA + CARLA 0.9.12

import os, csv, math
import numpy as np
from opencda.sustainability.utils import aerodynamic_force, rolling_resistance_force, gravity_force_on_slope


class SustainabilityMetrics:
    """
    Per-vehicle sustainability tracker:
    - integrates energy (J), regen potential (J), CO2 (g)
    - logs periodic CSV rows
    - computes eco-driving counters and final eco score
    """

    def __init__(self, vehicle, output_folder="cache/sustainability_logs",
                 vehicle_id=None, veh_params=None, model_type="auto", sample_hz=5):

        self.vehicle = vehicle
        self.id = vehicle_id or vehicle.id
        
        print(f"[SUST] Vehicle {self.vehicle.id} role:",
        self.vehicle.attributes.get("role_name"))
        
        self.veh_params = veh_params or {}

        # -------------------------------------------------------
        # VEHICLE PHYSICS PARAMETERS
        # -------------------------------------------------------
        try:
            phys = vehicle.get_physics_control()
            mass = phys.mass
        except Exception:
            mass = self.veh_params.get('mass', 1500.0)

        self.mass = self.veh_params.get('mass', mass)
        self.cd = self.veh_params.get('cd', 0.30)
        self.area = self.veh_params.get('area', 2.2)
        self.crr = self.veh_params.get('crr', 0.01)
        self.drivetrain_eff = self.veh_params.get('drivetrain_eff', 0.90)
        self.is_ev = self.veh_params.get('is_ev', False)

        if model_type == "ev":
            self.is_ev = True
        elif model_type == "ice":
            self.is_ev = False

        # -------------------------------------------------------
        # CUMULATIVE METRICS
        # -------------------------------------------------------
        self.energy_j = 0.0
        self.regen_j = 0.0
        self.co2_g = 0.0
        self.distance_m = 0.0

        # ECO DRIVING STATES
        self.harsh_accel = 0
        self.harsh_brake = 0
        self.idle_time_s = 0.0

        self.jerk_samples = []
        self.last_accel = 0.0
        self.last_speed = 0.0

        # -------------------------------------------------------
        # CSV LOGGING SETUP
        # -------------------------------------------------------
        os.makedirs(output_folder, exist_ok=True)
        self.csv_path = os.path.join(output_folder, f"vehicle_{self.id}_sustain.csv")
        self.sample_hz = sample_hz
        self._accum_dt = 0.0

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp','x','y','speed_m_s','long_accel_m_s2','jerk','power_w',
            'dt_s','cumulative_energy_j','cumulative_co2_g','regen_j','eco_score'
        ])

    def close(self):
        try:
            self.csv_file.close()
        except:
            pass

    # ---------------------------------------------------------
    # POWER + EMISSIONS
    # ---------------------------------------------------------
    def compute_power(self, speed, accel, slope=0.0):
        """Mechanical/electrical power in Watts."""
        F_aero = aerodynamic_force(self.cd, self.area, speed)
        F_roll = rolling_resistance_force(self.crr, self.mass)
        F_grav = gravity_force_on_slope(self.mass, slope)
        F_inertial = self.mass * accel

        F_total = F_aero + F_roll + F_grav + F_inertial
        P = F_total * speed

        # drivetrain efficiency
        if P >= 0:
            return P / max(self.drivetrain_eff, 1e-6)
        else:
            return P * self.drivetrain_eff

    def estimate_fuel_co2(self, power_w, dt_s):
        if power_w <= 0:
            return 0.0
        E_kwh = (power_w * dt_s) / 3.6e6
        liters = E_kwh / 8.9
        return liters * 2310.0

    def estimate_ev_co2(self, power_w, dt_s, grid_g_per_kwh=400.0):
        E_kwh = (power_w * dt_s) / 3.6e6
        return E_kwh * grid_g_per_kwh

    # ---------------------------------------------------------
    # MAIN UPDATE
    # ---------------------------------------------------------
    def update(self, snapshot, control=None, slope=0.0):

        dt = snapshot.timestamp.delta_seconds
        if dt <= 0:
            return

        vel = self.vehicle.get_velocity()
        acc = self.vehicle.get_acceleration()

        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # -----------------------------------------------------
        # LONGITUDINAL ACCELERATION (FIXED)
        # -----------------------------------------------------
        if speed > 0.1:
            accel = (vel.x*acc.x + vel.y*acc.y + vel.z*acc.z) / speed
        else:
            accel = 0.0

        # DISTANCE
        self.distance_m += speed * dt

        # POWER + ENERGY + CO2
        power_w = self.compute_power(speed, accel, slope)
        if power_w > 0:
            self.energy_j += power_w * dt
            if self.is_ev:
                self.co2_g += self.estimate_ev_co2(power_w, dt)
            else:
                self.co2_g += self.estimate_fuel_co2(power_w, dt)
        else:
            self.regen_j += abs(power_w) * dt * 0.5

        # -----------------------------------------------------
        # JERK CALCULATION (FIXED)
        # -----------------------------------------------------
        jerk = (accel - self.last_accel) / dt
        self.jerk_samples.append(jerk)
        self.last_accel = accel

        # -----------------------------------------------------
        # ECO DRIVING COUNTERS
        # -----------------------------------------------------
        if accel > 2.0:
            self.harsh_accel += 1
        if accel < -2.5:
            self.harsh_brake += 1

        if speed < 0.3 and (control is None or getattr(control, 'throttle', 0) < 0.1):
            self.idle_time_s += dt

        # -----------------------------------------------------
        # CSV LOGGING (sampled)
        # -----------------------------------------------------
        self._accum_dt += dt
        if self._accum_dt >= 1.0 / max(1, self.sample_hz):

            eco_score = self.compute_eco_score()
            loc = self.vehicle.get_location()

            self.csv_writer.writerow([
                snapshot.timestamp.elapsed_seconds,
                round(loc.x,3), round(loc.y,3),
                round(speed,3), round(accel,3),
                round(jerk,3),
                round(power_w,3),
                round(self._accum_dt,4),
                round(self.energy_j,3),
                round(self.co2_g,3),
                round(self.regen_j,3),
                round(eco_score,2)
            ])
            self.csv_file.flush()
            self._accum_dt = 0.0

    # ---------------------------------------------------------
    # ECO SCORE (NEW REALISTIC VERSION)
    # ---------------------------------------------------------
    def compute_eco_score(self):
        """
        Normalized eco score 0–100.
        Takes into account jerk, harsh events, and idle, normalized by distance.
        """

        dist_km = max(self.distance_m / 1000.0, 0.1)
        jerk_std = float(np.std(self.jerk_samples)) if len(self.jerk_samples) else 0.0

        # soft, realistic weights
        penalty = (
            (jerk_std * 1.2) +
            (self.harsh_accel * 0.08) +
            (self.harsh_brake * 0.12) +
            (self.idle_time_s * 0.01)
        ) / dist_km

        score = max(0.0, 100.0 - penalty)
        return score

    # ---------------------------------------------------------
    # FINAL RESULTS FOR SUMMARY.JSON
    # ---------------------------------------------------------
    def results(self):
        return {
            'vehicle_id': self.id,
            'energy_Wh': self.energy_j / 3600.0,
            'regen_Wh': self.regen_j / 3600.0,
            'co2_g': self.co2_g,
            'distance_m': self.distance_m,
            'harsh_accel': self.harsh_accel,
            'harsh_brake': self.harsh_brake,
            'idle_time_s': self.idle_time_s,
            'eco_score': self.compute_eco_score(),
        }
