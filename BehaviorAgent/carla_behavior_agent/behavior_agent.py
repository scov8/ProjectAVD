# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        self._overtake_threshold = 10
        self._change_lane_threshold = 4

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles

        gestisce il comportamento di tailgating, cambiando corsia se necessario e cerca di tenere in considerazione i veicoli che vengono da dietro.
        se stiamo andando troppo veloce rispetto al veicolo che ci sta dietro, il cambio di corsia lo possiamo fare.
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    # se non ci sono veicoli che ci ostacolano, cambia corsia, avvio la manovra di cambio corsia
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(
                        end_waypoint.transform.location, right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location, left_wpt.transform.location)

    def _overtake(self, to_overtake, vehicle_list):
        if self._behavior.overtake_doing == 0:
            print("vedo se posso fare il sorpasso")
            new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=180, lane_offset=-1)
            new_vehicle_state2, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit * 1.5), up_angle_th=40, lane_offset=-1)
            if not new_vehicle_state and not new_vehicle_state2:
                print("avvio il sorpasso")
                self._behavior.overtake_doing = 1
                self._behavior.overtake_counter = 55
                self.lane_change("left", other_lane_time=3, follow_direction=False)
                self._local_planner.set_speed(80)
        elif self._behavior.overtake_doing == 1 and self._behavior.overtake_counter == 0:
            print("vedo se posso finire il sorpasso")
            new_vehicle_state, _, _ = self._vehicle_obstacle_detected(to_overtake, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
            if not new_vehicle_state:
                print("finisco il sorpasso")
                self._behavior.overtake_doing = 0
                self._behavior.overtake_counter = 5
                self.lane_change("left", follow_direction=True)
                self._local_planner.set_speed(30)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        vehicle_list.sort(key=dist)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(
            waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]
        walker_list.sort(key=dist)

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)  # se quesro sensore influenza la cosa

        return walker_state, walker, distance

    def obstacle_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return obstacle_state: True if there is a object nearby, False if not
            :return obstacle: nearby obstacle
            :return distance: distance to nearby obstacle
        """

        obstacle_list = self._world.get_actors().filter("*static*")
        # funzione distanza, valuta la distanza tra il pedone e dove mi trovo
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        obstacle_list = [w for w in obstacle_list if dist(w) < 45]  # prendiamo quelli sotto i 10 mt
        obstacle_list.sort(key=dist)

        # in base a quello ceh dbb fare valutaimo in modo diverso _vehicle_obstacle_detected()
        if self._direction == RoadOption.CHANGELANELEFT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)  # se questo sensore influenza la cosa

        return obstacle_state, obstacle, distance

    def car_following_manager(self, vehicle, distance, debug=True):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        vehicle_list.sort(key=dist)

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        if (((vehicle_speed < (self._speed / 5)) or (vehicle_speed < 1.0)) and distance < 9.0) or self._behavior.overtake_doing == 1:
            print("potrei fare l'overtake")
            wpt = ego_vehicle_wp.get_left_lane()
            self._overtake(vehicle_list, vehicle_list)
            control = self._local_planner.run_step(debug=debug)

        # Under safety time distance, slow down.
        elif self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def obstacle_manager(self, obstacle, distance, debug=True):
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        vehicle_list.sort(key=dist)
        obstacle_list = self._world.get_actors().filter("*static*")
        obstacle_list = [v for v in obstacle_list if dist(v) < 45 and v.id != self._vehicle.id]
        obstacle_list.sort(key=dist)

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        control = self._local_planner.run_step(debug=debug)

        if distance <= self._behavior.braking_distance:
            self._overtake(obstacle_list, vehicle_list)
            control = self._local_planner.run_step(debug=debug)

        return control

    def run_step(self, debug=True):  # il debug era false
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl

        chiamato ad ogni run step del sistema, gestisce semafori, stop, pedoni ect
        """
        self._update_information()

        control = None

        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        if self._behavior.overtake_counter > 0:
            self._behavior.overtake_counter -= 1

        # prende le info del veicolo
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        # vede se esiste in un certo range un semaforo e vede se è rosso,si ferma e si salva che è rosso al semaforo.
        if self.traffic_light_manager():
            return self.emergency_stop()  # se è rosso si ferma

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(
            ego_vehicle_wp)  # lo considero fermandomi

        # definisco se esiste un pednone che impatta con il veiolo
        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close al pedone.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.1.2: Obstacle avoidance behaviors
        obstacle_state, obstacle, distance = self.obstacle_avoid_manager(ego_vehicle_wp)

        if obstacle_state:
            distance = distance - max(
                obstacle.bounding_box.extent.y, obstacle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance and self._speed != 0:
                return self.emergency_stop()
            elif self._speed == 0:
                self.obstacle_manager(obstacle, distance)
                #pass
            else:
                # faccio rallentare la macchina
                target_speed = self._speed - (self._behavior.speed_decrease-3)
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                # se il veicolo non è molto vicino posso pensare di seguirlo
                control = self.car_following_manager(vehicle, distance)
        
        elif self._behavior.overtake_doing == 1:
            ego_vehicle_loc = self._vehicle.get_location()
            ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
            vehicle_list = self._world.get_actors().filter("*vehicle*")
            def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
            vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
            vehicle_list.sort(key=dist)
            self._overtake(vehicle_list, vehicle_list)
            control = self._local_planner.run_step(debug=debug)

        # 3: Intersection behavior
        # è una fregatura, ci dice se stiamo nellincrocio ma la gestione non è diversa da quella del behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            # se non ci sono pedoni, nemmeno macchine che ci stanno davanti, allora procediamo normalmente
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            # adattiamo il local planner a seguire la nostra velocità, dentro al local ci sono i controllori e cambia la vel
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control  # una volta che abbiamo il controllo, lo ritornaimo

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
