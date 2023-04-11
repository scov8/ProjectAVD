# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import IntEnum
from collections import deque
import random

import carla
from controller import VehicleController
from misc import draw_waypoints, get_speed


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):

    FPS = 20

    def __init__(self, agent):

        self._vehicle = agent.vehicle

        self._map = agent.vehicle.get_world().get_map()
        self._target_speed = None
        self.sampling_radius = None
        self._min_distance = 3
        self._current_distance = None
        self.target_road_option = None
        self._vehicle_controller = None
        self._global_plan = None
        self._pid_controller = None
        self.waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
        self.args_lat_city_dict = {
            'K_P': 0.58,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1.0 / self.FPS}
        self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
        self.args_long_city_dict = {
            'K_P': 0.15,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

    def get_incoming_waypoint_and_direction(self, steps=3):

        if len(self.waypoints_queue) > steps:
            return self.waypoints_queue[steps]

        else:

            try:
                wpt, direction = self.waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:

                return None, RoadOption.VOID

        return None, RoadOption.VOID

    def set_speed(self, speed):

        self._target_speed = speed

    def set_global_plan(self, current_plan, clean=True):

        print('set_global_plan called')

        for elem in current_plan:

            self.waypoints_queue.append(elem)

        if clean:

            self._waypoint_buffer.clear()

            for _ in range(self._buffer_size):

                if self.waypoints_queue:

                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())

                else:
                    break

        self._global_plan = True

    def run_step(self, target_speed=None, debug=False):

        print('run_step  _local_planner called')

        if target_speed is not None:

            self._target_speed = target_speed

        else:

            self.target_speed = self.vehicle.get_speed_limit()

        if (len(self.waypoints_queue) == 0):

            control = carla.VehicleControl()

            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        if not self._waypoint_buffer:

            for i in range(self._buffer_size):

                if self.waypoints_queue:

                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())

                else:

                    break

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())

        self.target_waypoint, self.target_road_option = self._waypoint_buffer[0]

        if target_speed > 50:

            args_lat = self.args_lat_hw_dict
            args_long = self.args_long_hw_dict

        else:

            args_lat = self.args_lat_city_dict
            args_long = self.args_long_city_dict

        self._pid_controller = VehiclePIDController(
            self._vehicle, args_lateral=args_lat, args_longitudinal=args_long)

        control = self._pid_controller.run_step(
            self._target_speed, self.target_waypoint)

        vehicle_transfrom = self._vehicle.get_transform()

        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):

            if distance_vehicle(waypoint, vehicle_transfrom) < self._min_distance:

                max_index = i

        if max_index >= 0:

            for i in range(max_index+1):

                self._waypoint_buffer.popleft()

        return control
