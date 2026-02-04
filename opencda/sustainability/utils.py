# utils.py
# Physics and helper functions for sustainability calculations in CARLA / OpenCDA
# Place this file at: C:\Users\goton\OpenCDA\opencda\sustainability\utils.py

import math

# Physical constants (SI)
G = 9.81                      # m/s^2 gravity
RHO_AIR = 1.225               # kg/m^3, air density at sea level

# Default vehicle parameters (can be overridden per-vehicle)
DEFAULT_CD = 0.30             # drag coefficient (typical passenger car)
DEFAULT_FRONTAL_AREA = 2.2    # m^2 frontal area
DEFAULT_CRR = 0.01            # rolling resistance coefficient
DEFAULT_DRIVETRAIN_EFF = 0.9  # drivetrain efficiency (fraction)
WATT_TO_WH = 1.0 / 3600.0     # convert Watt-seconds (J) to Watt-hours (Wh)

# --- Helper physics functions ---

def aerodynamic_force(cd: float, area: float, speed: float) -> float:
    """
    Aerodynamic drag force (N): 0.5 * rho * cd * A * v^2
    """
    return 0.5 * RHO_AIR * cd * area * (speed ** 2)


def rolling_resistance_force(crr: float, mass: float) -> float:
    """
    Rolling resistance force (N): crr * m * g
    """
    return crr * mass * G


def gravity_force_on_slope(mass: float, slope: float) -> float:
    """
    Component of gravity along a slope (N).
    slope is the road grade (rise/run), e.g. 0.01 for 1% grade.
    """
    return mass * G * slope


def kinetic_energy(mass: float, speed: float) -> float:
    """
    Kinetic energy in Joules: 0.5 * m * v^2
    """
    return 0.5 * mass * (speed ** 2)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Helper: safe division returning default if denominator is zero.
    """
    try:
        return a / b
    except Exception:
        return default
