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

from misc import get_speed, positive, is_within_distance, compute_distance, get_steering


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
        self._steer = 0
        self._speed_limit = 0
        self._direction = None
        self._vehicle_heading = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._prev_direction = RoadOption.LANEFOLLOW
        self._overtaking = False
        self._overtaking_obj = False
        self._ending_overtake = False
        self._destination_waypoint = None
        self._restringimento = False

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
        self._steer = get_steering(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._vehicle_heading = self._vehicle.get_transform().rotation.yaw
        self._prev_direction = self._direction
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        # Save final destination waypoint.
        if self._destination_waypoint is None:
            if not self._overtaking:
                self._destination_waypoint = self._local_planner._waypoints_queue[-1][0]
            if not self._overtaking_obj:
                self._destination_waypoint = self._local_planner._waypoints_queue[-1][0]

    def _other_lane_occupied(self, ego_loc, distance, check_behind=False):
        if self._overtaking_obj:
            vehicle_list = self._world.get_actors().filter("*static*")
        else:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v, w): return v.get_location().distance(
            w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x
        vehicle_list = [v for v in vehicle_list if dist(
            v, self._vehicle) < distance and v.id != self._vehicle.id]

        if check_behind is False:
            vehicle_state, vehicle, distance = self._vehicle_detected_other_lane(
                vehicle_list, distance, up_angle_th=90)
            if vehicle_state:
                print("OTHER LANE OCCUPATA DA: " + str(vehicle))
                return True
            return False
        else:
            vehicle_state_ahead, vehicle_ahead, distance_ahead = self._vehicle_detected_other_lane(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, check_rear=True)
            vehicle_state_behind, vehicle_behind, distance_behind = self._vehicle_detected_other_lane(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), low_angle_th=90, up_angle_th=135)
            if vehicle_state_ahead and vehicle_state_behind:
                print(
                    f"OTHER LANE OCCUPATA AHEAD: {vehicle_ahead} e BEHIND: {vehicle_behind} distanti {dist(vehicle_ahead, vehicle_behind)}")
                return dist(vehicle_ahead, vehicle_behind) <= self._vehicle.bounding_box.extent.x * 2 + 5
            elif vehicle_state_ahead:
                print("OTHER LANE OCCUPATA AHEAD DA: " + str(vehicle_ahead))
                return True
            elif vehicle_state_behind:
                print(
                    f"VEICOLO BEHIND {vehicle_behind} è lontano {distance_behind}")
                return distance_behind < self._vehicle.bounding_box.extent.x * 2.5
            return False

    def _other_lane_occupied_bis(self, ego_loc, distance):
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v, w): return v.get_location().distance(
            w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x
        vehicle_list = [v for v in vehicle_list if dist(
            v, self._vehicle) < distance and v.id != self._vehicle.id]

        vehicle_state, vehicle, distance = self._vehicle_detected_other_lane(
            vehicle_list, distance, up_angle_th=90)
        if vehicle_state:
            print("OTHER LANE OCCUPATA DA: " + str(vehicle))
            return True, vehicle
        return False, None

    def _is_slow(self, vehicle):
        vel = vehicle.get_velocity().length()
        acc = vehicle.get_acceleration().length()
        return acc <= 1.0 and vel < 3

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
            if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
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
                    self.set_destination(
                        end_waypoint.transform.location, left_wpt.transform.location)

    def _lane_invasion(self, ego_vehicle, other_vehicle, ego_loc):
        ego_wp = self._map.get_waypoint(ego_loc, project_to_road=False)
        other_loc = other_vehicle.get_location()
        other_lane_wp = self._map.get_waypoint(other_loc)

        other_offset = other_lane_wp.transform.location.distance(
            other_vehicle.get_location())
        other_extent = other_vehicle.bounding_box.extent.y
        lane_width = other_lane_wp.lane_width
        free_space_on_one_side = lane_width / 2 - other_extent

        if other_offset > free_space_on_one_side:
            print('other_offset is larger thant free space on one side')
            return True, other_offset - free_space_on_one_side
        return False, 0

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
        def dist(v): return v.get_location().distance(
            waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(
            v) < 45 and v.id != self._vehicle.id]
        vehicle_list.sort(key=dist)
        print("VEHICLE LIST: " + str(vehicle_list))

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self._speed > 10 and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        print("vehicle_state: ", vehicle_state,
              "vehicle: ", vehicle, "distance: ", distance)
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

        if walker_list == []:
            return False, None, None

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)  # se quesro sensore influenza la cosa

        print("walker_state: ", walker_state,
              "walker: ", walker, "distance: ", distance)
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
        def dist(w): return w.get_location().distance(
            waypoint.transform.location)
        obstacle_list = [w for w in obstacle_list if dist(
            w) < 45]  # prendiamo quelli sotto i 10 mt
        obstacle_list.sort(key=dist)

        if obstacle_list == []:
            return False, None, None

        # in base a quello ceh dbb fare valutaimo in modo diverso _vehicle_obstacle_detected()
        if self._direction == RoadOption.CHANGELANELEFT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=40)  # se questo sensore influenza la cosa

        print("obstacle_state", obstacle_state,
              "obstacle", obstacle, "distance", distance)
        return obstacle_state, obstacle, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / \
            np.nextafter(0., 1.)

        print("VEICOLO DAVANTI. Distance: ", distance, "Velocità ego: ",
              self._speed, "Velocità veicolo davanti: ", vehicle_speed)
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(
                    vehicle_speed - self._behavior.speed_decrease), self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist
            ])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed), self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist
            ])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        else:
            target_speed = min(
                [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
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

        # prende le info del veicolo
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        # vede se esiste in un certo range un semaforo e vede se è rosso,si ferma e si salva che è rosso al semaforo.
        if self.traffic_light_manager():
            return self.emergency_stop()  # se è rosso si ferma

        # 2.3: Lane Invasion (degli altri)
        vehicle_state_invasion, vehicle_invasion = self._other_lane_occupied_bis(
            ego_vehicle_loc, distance=70)
        if vehicle_state_invasion:
            invasion_state, offset_invasion = self._lane_invasion(
                self._vehicle, vehicle_invasion, ego_vehicle_loc)
            if invasion_state:
                print('LANE INVASION: TRUE, SO DO EMERGENCY STOP')
                self.stay_on_the_right(ego_vehicle_wp, offset_invasion-2.3, 2)
                # self._local_planner.set_lat_offset(offset_invasion) # mio
                self._restringimento = True
                target_speed = min(
                    [self._behavior.max_speed, self._speed_limit]) - (self._behavior.speed_decrease * 3)
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
        elif self._restringimento:
            print('LANE INVASION: FALSE')
            self._local_planner.set_lat_offset(0.0)
            route_trace = self.trace_route(
                ego_vehicle_wp, self._destination_waypoint)
            self._local_planner.set_global_plan(route_trace,  clean_queue=True)
            self._restringimento = False

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(
            ego_vehicle_wp)  # lo considero fermandomi

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close al pedone.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.1.2: Obstacle avoidance behaviors
        obstacle_state, obstacle, distance = self.obstacle_avoid_manager(
            ego_vehicle_wp)

        if obstacle_state:
            distance = distance - max(obstacle.bounding_box.extent.y, obstacle.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if self._speed < 0.01:
                if ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken:
                    if not self._overtaking_obj and self._direction == RoadOption.LANEFOLLOW:
                        if not self._other_lane_occupied(ego_vehicle_loc, distance=70):
                            if self.lane_change("left", self._vehicle_heading, 0, 2, 2):
                                print("cambio corsia a sinistra per ostacolo")
                                self._overtaking_obj = True
                                target_speed = max(
                                    [self._behavior.max_speed, self._speed_limit])
                                self._local_planner.set_speed(target_speed)
                                control = self._local_planner.run_step(
                                    debug=debug)
                                return control
                # pass
            elif distance < self._behavior.braking_distance and self._speed > 0.01 and not self._overtaking_obj:
                print("sto frenando per ostacolo: EMERGENCY STOP")
                return self.emergency_stop()
            elif distance < 13 and self._speed > 0.01 and not self._overtaking_obj:
                print("sto frenando per ostacolo: SOFT STOP")
                return self.soft_stop()
            elif distance < 30 and self._speed > 0.01 and not self._overtaking_obj:
                print("sto frenando per ostacolo: NO THROTTLE")
                return self.no_throttle()

        # 2.2.1: overtake behavior
        if self._ending_overtake:
            print("sto terminando sorpasso")
            if not self._local_planner.has_incoming_waypoint():
                self._ending_overtake = False
                self._overtaking = False
                self._overtaking_obj = False
                route_trace = self.trace_route(
                    ego_vehicle_wp, self._destination_waypoint)
                self._local_planner.set_global_plan(route_trace, True)
                print(
                    f"SORPASSO TERMINATO, deque len: {len(self._local_planner._waypoints_queue)}")
            target_speed = min([self._behavior.max_speed, self._speed_limit])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            return control
        elif self._overtaking or self._overtaking_obj:
            print("sorpasso in corso...")
            if not self._local_planner.has_incoming_waypoint():
                if not self._other_lane_occupied(ego_vehicle_loc, 15, check_behind=True):
                    print("RIENTRO")
                    if self.lane_change("left", self._vehicle_heading, 0, 2, 2):
                        self._ending_overtake = True
                else:
                    self.lane_change("left", self._vehicle_heading, 1, 0, 0)

            target_speed = min([self._behavior.max_speed, self._speed_limit])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            return control

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(
            ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            if ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken:
                if not self._overtaking and self._direction == RoadOption.LANEFOLLOW:
                    if self._is_slow(vehicle):
                        vehicle_list = self._world.get_actors().filter("*vehicle*")
                        def dist(v, w): return v.get_location().distance(
                            w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x
                        vehicle_list = [v for v in vehicle_list if dist(
                            v, self._vehicle) < 30 and v.id != self._vehicle.id]
                        new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                            self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=180, lane_offset=-1)
                        new_vehicle_state2, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                            self._behavior.min_proximity_threshold, self._speed_limit), low_angle_th=90, up_angle_th=180, lane_offset=-1)
                        if not new_vehicle_state and not new_vehicle_state2:
                            if not self._other_lane_occupied(ego_vehicle_loc, distance=70) and not self._overtaking:
                                if self.lane_change("left", self._vehicle_heading, 0, 2, 2):
                                    self._overtaking = True
                                    target_speed = max(
                                        [self._behavior.max_speed, self._speed_limit])
                                    self._local_planner.set_speed(target_speed)
                                    control = self._local_planner.run_step(
                                        debug=debug)
                                    return control

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                # se il veicolo non è molto vicino posso pensare di seguirlo
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        # è una fregatura, ci dice se stiamo nellincrocio ma la gestione non è diversa da quella del behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min(
                [self._behavior.max_speed, self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # se sto andando molto veloce e con lo sterzo ho un valore sopra al 0.5, allora rallento
        elif self._speed > 45 and self._steer > 90:
            print("velocità troppo alta, rallentooooooooooooooooooooooooooooooo")
            return self.no_throttle()

        # 4: Normal behavior
        else:
            # se non ci sono pedoni, nemmeno macchine che ci stanno davanti, allora procediamo normalmente
            target_speed = min(
                [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
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

    def soft_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.2
        control.hand_brake = False
        return control

    def no_throttle(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        return control
