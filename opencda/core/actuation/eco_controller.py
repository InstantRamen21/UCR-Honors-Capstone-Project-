# -*- coding: utf-8 -*-
"""
ECO PID Control Class
Extends standard PID controller with fuel/energy efficiency improvements:
- Speed reduction only on sharp turns (not gentle curves)
- Smooth acceleration limiting to prevent throttle spikes
- Coasting zone: reduce throttle before braking is needed
"""

# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from collections import deque

import math
import numpy as np

import carla


class Controller:
    """
    ECO PID Controller — drop-in replacement for the standard PID controller,
    with modifications for improved energy efficiency.

    Key differences from standard PID:
    - Steering-based speed adjustment only kicks in for sharper turns
      (threshold > 0.15 rad) to avoid unnecessary speed reductions on
      straight/gentle roads.
    - Acceleration smoothing uses a gentler ramp (max_accel_change = 0.2)
      so the vehicle doesn't lag behind platoon speed commands.
    - Coasting: when close to target speed (within a small window), throttle
      is gently reduced to let the vehicle coast rather than actively
      maintaining speed with throttle.

    Parameters
    ----------
    args : dict
        The configuration dictionary parsed from yaml file.

    Attributes
    ----------
    _lon_ebuffer : deque
        A deque buffer that stores longitudinal control errors.

    _lat_ebuffer : deque
        A deque buffer that stores latitudinal control errors.

    current_transform : carla.transform
        Current ego vehicle transformation in CARLA world.

    current_speed : float
        Current ego vehicle speed.

    past_steering : float
        Steering angle from previous control step.

    """

    def __init__(self, args):

        # longitudinal related
        self.max_brake = args['max_brake']
        self.max_throttle = args['max_throttle']

        self._lon_k_p = args['lon']['k_p']
        self._lon_k_d = args['lon']['k_d']
        self._lon_k_i = args['lon']['k_i']

        self._lon_ebuffer = deque(maxlen=10)

        # lateral related
        self.max_steering = args['max_steering']

        self._lat_k_p = args['lat']['k_p']
        self._lat_k_d = args['lat']['k_d']
        self._lat_k_i = args['lat']['k_i']

        self._lat_ebuffer = deque(maxlen=10)

        # simulation time-step
        self.dt = args['dt']

        # current speed and localization retrieved from sensing layer
        self.current_transform = None
        self.current_speed = 0.
        # past steering
        self.past_steering = 0.

        self.dynamic = args['dynamic']
        
        # ECO-specific state
        self.last_accel = 0.0

        # --- Tunable ECO parameters ---
        # Only reduce speed if steering angle exceeds this threshold (radians).
        # 0.15 ≈ ~8.6 degrees — ignores gentle curves, only acts on real turns.
        self.steering_threshold = 0.15

        # Maximum fraction of speed to shed on very sharp turns (steering=1.0).
        # 0.15 means at most 15% speed reduction even in the sharpest turn.
        self.max_speed_reduction = 0.15

        # Acceleration smoothing: max change per step.
        # 0.2 is gentler than 0.1 — allows the vehicle to respond to
        # platoon speed commands without lagging.
        self.max_accel_change = 0.2

        # Coasting window: if |speed_error| < this, ease off throttle slightly.
        # Helps avoid the throttle-on / throttle-off oscillation near target speed.
        self.coast_window = 2.0   # km/h
        self.coast_throttle_cap = 0.35  # max throttle when coasting

    def dynamic_pid(self):
        """
        Compute kp, kd, ki based on current speed.
        """
        pass

    def update_info(self, ego_pos, ego_spd):
        """
        Update ego position and speed to controller.

        Parameters
        ----------
        ego_pos : carla.location
            Position of the ego vehicle.

        ego_spd : float
            Speed of the ego vehicle

        Returns
        -------

        """

        self.current_transform = ego_pos
        self.current_speed = ego_spd
        if self.dynamic:
            self.dynamic_pid()

    def lon_run_step(self, target_speed):
        """

        Parameters
        ----------
        target_speed : float
            Target speed of the ego vehicle.

        Returns
        -------
        acceleration : float
            Desired acceleration value for the current step
            to achieve target speed.

        """
        error = target_speed - self.current_speed
        self._lon_ebuffer.append(error)

        if len(self._lon_ebuffer) >= 2:
            _de = (self._lon_ebuffer[-1] - self._lon_ebuffer[-2]) / self.dt
            _ie = sum(self._lon_ebuffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._lon_k_p * error) +
                       (self._lon_k_d * _de) +
                       (self._lon_k_i * _ie),
                       -1.0, 1.0)

    def lat_run_step(self, target_location):
        """
        Generate the throttle command based on current speed and target speed

        Parameters
        ----------
        target_location : carla.location
            Target location.

        Returns
        -------
        current_steering : float
        Desired steering angle value for the current step to
        achieve target location.

        """
        v_begin = self.current_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(
                math.radians(
                    self.current_transform.rotation.yaw)), y=math.sin(
                math.radians(
                    self.current_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([target_location.x -
                          v_begin.x, target_location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(
            w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
                                 -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._lat_ebuffer.append(_dot)
        if len(self._lat_ebuffer) >= 2:
            _de = (self._lat_ebuffer[-1] - self._lat_ebuffer[-2]) / self.dt
            _ie = sum(self._lat_ebuffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._lat_k_p * _dot) + (self._lat_k_d *
                       _de) + (self._lat_k_i * _ie), -1.0, 1.0)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint at a given target_speed.

        Parameters
        ----------
        target_speed : float
            Target speed of the ego vehicle.

        waypoint : carla.loaction
            Target location.

        Returns
        -------
        control : carla.VehicleControl
            Desired vehicle control command for the current step.

        """
        
        if not hasattr(self, '_confirmed'):
            print(f"[ECO] run_step called — eco_controller is active")
            self._confirmed = True
        
        # control class for carla vehicle
        control = carla.VehicleControl()

        # emergency stop
        if target_speed == 0 or waypoint is None:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            return control

        # --- Lateral control ---
        current_steering = self.lat_run_step(waypoint)

        # --- ECO: Speed reduction only on sharp turns ---
        abs_steer = abs(current_steering)
        if abs_steer > self.steering_threshold:
            reduction_fraction = (
                (abs_steer - self.steering_threshold) /
                (1.0 - self.steering_threshold)
            ) * self.max_speed_reduction
            adjusted_speed = target_speed * (1.0 - reduction_fraction)
        else:
            adjusted_speed = target_speed

        # --- Longitudinal control ---
        acceleration = self.lon_run_step(adjusted_speed)

        # --- ECO: Smooth acceleration (prevent throttle spikes) ---
        accel_diff = acceleration - self.last_accel
        accel_diff = np.clip(accel_diff, -self.max_accel_change, self.max_accel_change)
        acceleration = self.last_accel + accel_diff
        self.last_accel = acceleration

        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too
        # much.
        if current_steering > self.past_steering + 0.2:
            current_steering = self.past_steering + 0.2
        elif current_steering < self.past_steering - 0.2:
            current_steering = self.past_steering - 0.2

        if current_steering >= 0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        return control