# Autonomous Vehicle Driving Project.
# Copyright (C) 2023 - All Rights Reserved
# Group:
#   Faiella Ciro              0622701816  c.faiella8@studenti.unisa.it
#   Giannino Pio Roberto      0622701713	p.giannino@studenti.unisa.it
#   Scovotto Luigi            0622701702  l.scovotto1@studenti.unisa.it
#   Tortora Francesco         0622701700  f.tortora21@studenti.unisa.it

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import math
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
        self._speed = 0             # speed of the ego vehicle
        self._steer = 0             # steering wheel angle of the ego vehicle
        self._speed_limit = 0       # speed limit of the road the ego vehicle
        self._direction = None      # direction the ego vehicle is following
        self._vehicle_heading = None        # heading of the ego vehicle
        self._incoming_direction = None     # direction of the incoming waypoint
        self._incoming_waypoint = None      # incoming waypoint
        self._min_speed = 5                 # minimum speed of the ego vehicle
        self._behavior = None               # behavior of the ego vehicle
        self._sampling_resolution = 4.5     # sampling resolution of the global router
        self._prev_direction = RoadOption.LANEFOLLOW    # previous direction of the ego vehicle
        self._overtaking_vehicle = False  # flag to indicate overtaking vehicle
        self._overtaking_obj = False      # flag to indicate overtaking object
        self._ending_overtake = False     # flag to indicate the end of overtaking
        self._destination_waypoint = None # final destination waypoint
        self._shrinkage = False           # shrinkage flag
        self._waypoints_queue_copy = None # copy of the waypoints queue
        self._d_max = 8                   # maximum distance to check for overtaking
        self._distance_to_over = 75       # distance to overtake
        self._distance_to_overtake_obj = 80    # distance to overtake
        self._n_vehicle = 0                 # number of vehicles in front of the ego vehicle
        self._stay_at_stop_counter = 15     # counter to stay at stop

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious() # set the behavior to cautious

        elif behavior == 'normal':
            self._behavior = Normal() # set the behavior to normal

        elif behavior == 'aggressive':
            self._behavior = Aggressive() # set the behavior to aggressive

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)      # get the speed of the ego vehicle
        self._steer = get_steering(self._vehicle)   # get the steering wheel angle of the ego vehicle
        self._speed_limit = self._vehicle.get_speed_limit() # get the speed limit of the road the ego vehicle is in
        self._local_planner.set_speed(self._speed_limit)    # set the speed of the ego vehicle
        self._vehicle_heading = self._vehicle.get_transform().rotation.yaw  # get the heading of the ego vehicle
        self._prev_direction = self._direction  # get the previous direction of the ego vehicle
        self._direction = self._local_planner.target_road_option    # get the direction the ego vehicle is following
        if self._direction is None: # if the direction is None, we set it to LANEFOLLOW
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10) # set the look ahead steps based on the speed limit of the road the ego vehicle is in

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=self._look_ahead_steps) # get the incoming waypoint and direction
        # if the incoming direction is None, we set it to LANEFOLLOW
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        # Save final destination waypoint.
        # if the final destination waypoint is None and we are not doing an overtake maneuver, we set it to the final waypoint of the waypoints queue
        if self._destination_waypoint is None:
            if not self._overtaking_vehicle:
                self._destination_waypoint = self._local_planner._waypoints_queue[-1][0]
            if not self._overtaking_obj:
                self._destination_waypoint = self._local_planner._waypoints_queue[-1][0]

    def _other_lane_occupied(self, distance, check_behind=False):
        """
        This method returns True if the other lane is occupied by a vehicle.
            :param distance: distance to look ahead for other vehicles
            :param check_behind: if True check also for vehicles behind the ego vehicle, otherwise check for vehicles ahead of the ego vehicle
            :return True if the other lane is occupied by a vehicle, False otherwise
        """

        # Get all the actors present in the world, in case the overtake is on a static object we get all the static actors, otherwise we get all the vehicles.
        if self._overtaking_obj:
            vehicle_list = self._world.get_actors().filter("*static*")
        else:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v, w): return v.get_location().distance(w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x # return distance between two vehicles
        vehicle_list = [v for v in vehicle_list if dist(v, self._vehicle) < distance and v.id != self._vehicle.id] # filter vehicles within distance and not the ego vehicle

        # If the flag check_behind is False, we check the ahead vehicle
        if check_behind is False:
            vehicle_state, vehicle, distance = self._vehicle_detected_other_lane(vehicle_list, distance, up_angle_th=90) # check for vehicles in the other lane
            # If a vehicle is detected in the other lane, we return True, otherwise we return False.
            if vehicle_state:
                print("OTHER LANE OCCUPATA DA: ", str(vehicle), "CON DISTANZA: ", dist(vehicle, self._vehicle))
                return True
            return False
        else:
            vehicle_state_ahead, vehicle_ahead, distance_ahead = self._vehicle_detected_other_lane(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, check_rear=True)      # check for vehicles in the other lane ahead of the ego vehicle
            vehicle_state_behind, vehicle_behind, distance_behind = self._vehicle_detected_other_lane(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), low_angle_th=90, up_angle_th=135)  # check for vehicles in the other lane behind of the ego vehicle

            # If a vehicle is detected in the other lane on ahead and behind
            if vehicle_state_ahead and vehicle_state_behind:
                print(f"OTHER LANE OCCUPATA AHEAD: {vehicle_ahead} e BEHIND: {vehicle_behind} distanti {dist(vehicle_ahead, vehicle_behind)}")
                return dist(vehicle_ahead, vehicle_behind) <= self._vehicle.bounding_box.extent.x * 2 + 5
            # If a vehicle is detected in the other lane on ahead return True
            elif vehicle_state_ahead:
                print("OTHER LANE OCCUPATA AHEAD DA: " + str(vehicle_ahead))
                return True
            # If a vehicle is detected in the other lane behind return True
            elif vehicle_state_behind:
                print(f"VEICOLO BEHIND {vehicle_behind} è lontano {distance_behind}")
                return distance_behind - 3 < self._vehicle.bounding_box.extent.x * 2.5
            return False

    def _other_lane_occupied_lane_invasion(self, distance):
        """
        This method returns True if the other lane is occupied by a vehicle.
            :param distance: distance to look ahead for other vehicles
            :return True if the other lane is occupied by a vehicle, False otherwise
            :return the distance between the ego vehicle and the vehicle in the other lane 
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*") # get all the vehicles in the world
        def dist(v, w): return v.get_location().distance(w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x # distance between two vehicles
        vehicle_list = [v for v in vehicle_list if dist(v, self._vehicle) < distance and v.id != self._vehicle.id] # filter vehicles within distance and not the ego vehicle

        vehicle_state, vehicle, distance = self._vehicle_detected_other_lane(vehicle_list, distance, up_angle_th=90) # check for vehicles in the other lane
        if vehicle_state: # if a vehicle is detected in the other lane, we return True and the distance between the ego vehicle and the vehicle in the other lane
            print("OTHER LANE OCCUPATA DA: " + str(vehicle))
            return True, vehicle
        return False, None

    def _is_slow(self, vehicle):
        """
        This method returns True if the vehicle is going slow.
            :param vehicle: the vehicle to check
            :return True if the vehicle is going slow, False otherwise
        """
        vel = vehicle.get_velocity().length()       # get the velocity of the vehicle
        acc = vehicle.get_acceleration().length()   # get the acceleration of the vehicle
        print("VELOCITA VEICOLO: ", vel, "ACCELERAZIONE: ", acc)
        return acc <= 1.0 and vel < 3 # if the acceleration is low and the velocity is low, we return True, otherwise we return False
    
    def _iam_stuck(self, waypoint):
        """
        This method check if there are vehicles in front of the ego vehicle.
            :param waypoint: the waypoint of the ego vehicle
            :return True, if there are vehicles in front of the ego vehicle, otherwise return False
            :return the number of vehicles in front of the ego vehicle
            :return the total distance between the ego vehicle and the last vehicle in front of it
            :return the maximum distance between two vehicles in front of the ego vehicle
        """
        ego_location = waypoint.transform.location      # get the location of the ego vehicle
        ego_wpt = self._map.get_waypoint(ego_location)  # get the waypoint of the ego vehicle
        vehicle_list = self._world.get_actors().filter("*vehicle*") # get all the vehicles in the world
        def dist(v): return v.get_location().distance(waypoint.transform.location)  # distance between the waypoint and the location of the vehicle
        vehicle_list = [v for v in vehicle_list if dist(v) < 60 and v.id != self._vehicle.id and (self._map.get_waypoint(v.get_transform().location).lane_id == ego_wpt.lane_id or (abs(self._map.get_waypoint(v.get_transform().location).lane_id) == abs(ego_wpt.lane_id) + 1))] # filter vehicles within 60 meters from the waypoint and not the ego vehicle and in the same lane or in the lane next to the ego vehicle
        vehicle_list.sort(key=dist) # sort the list of vehicles by distance
        vehicle_yaw = math.radians(self._vehicle.get_transform().rotation.yaw)  # get the yaw of the ego vehicle

        # remove vehicles that are behind the ego vehicle
        for v in vehicle_list:
            v_location = v.get_transform().location # get the location of the vehicle
            v_direction = math.atan2(v_location.y - ego_location.y, v_location.x - ego_location.x)  # get the direction of the vehicle
            relative_direction = abs(math.degrees(vehicle_yaw - v_direction))   # get the relative direction between the ego vehicle and the vehicle
            # if the relative direction is greater than 90 degrees, we remove the vehicle from the list
            if relative_direction >= 90:
                vehicle_list.remove(v)
        
        distance = 0    # initialize the distance between the ego vehicle and the last vehicle in front of it
        d_max=10        # initialize the maximum distance between two vehicles in front of the ego vehicle
        v_list =[]      # initialize the list of vehicles in front of the ego vehicle
        i=0             # initialize the index of the list of vehicles in front of the ego vehicle
        # create a list of vehicles that are in maximum 15 meters from each other
        for i in range (len(vehicle_list)-1):
            print("i: ", i, "len: ", len(vehicle_list))
            v1_location = vehicle_list[i].get_transform().location      # get the location of the first vehicle
            v2_location = vehicle_list[i+1].get_transform().location    # get the location of the second vehicle
            v_distance = math.sqrt((v2_location.x - v1_location.x)**2 + (v2_location.y - v1_location.y)**2) # distance between the first vehicle and the second vehicle
            #print("VEICOLO: ", v, "DISTANZA: ", v.get_location().distance(self._vehicle.get_location()))
            print("VEICOLO: ", vehicle_list[i], "DISTANZA: ", v_distance)
            print("i: ", i, "len: ", len(vehicle_list))
            # if the distance between the first vehicle and the second vehicle is less than 15 meters, we add the first vehicle to the list of vehicles in front of the ego vehicle
            if v_distance < 15:
                v_list.append(vehicle_list[i])  # add the first vehicle to the list of vehicles in front of the ego vehicle
                distance = v2_location.distance(self._vehicle.get_location())   # update the distance between the ego vehicle and the last vehicle in front of it
                # update the maximum distance between two vehicles in front of the ego vehicle
                if v_distance > d_max:
                    d_max = v_distance
            else:
                break
        v_list.append(vehicle_list[i])# add the last vehicle to the list of vehicles in front of the ego vehicle
        # return True, the number of vehicles in front of the ego vehicle, the total distance between the ego vehicle and the last vehicle in front of it, and the maximum distance between two vehicles in front of the ego vehicle
        if len(v_list) == 1 or len(v_list) == 0:
            print("I AM STUCK - VEICOLI DAVANTI A ME: ", 1, "DISTANZA TOTALE: ",65, "DISTANZA MASSIMA: ", 8)
            return True, 1, 65, 8
        elif len(v_list) == 2:
            print("I AM STUCK - VEICOLI DAVANTI A ME: ", 2, "DISTANZA TOTALE: ",65, "DISTANZA MASSIMA: ", 8)
            return True, 2, 65, 8
        else:
            print("I AM STUCK - VEICOLI DAVANTI A ME: ", len(v_list), "DISTANZA TOTALE: ",distance*3, "DISTANZA MASSIMA: ", d_max+1)
            return True, len(v_list), max(80, distance*3), d_max+1
    
    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        :return affected: True if the ego vehicle is affected by a red light, False otherwise
        """
        actor_list = self._world.get_actors() # get all the actors in the world
        lights_list = actor_list.filter("*traffic_light*") # get all the traffic lights in the world
        affected, _ = self._affected_by_traffic_light(lights_list) # check if the ego vehicle is affected by a traffic light

        return affected # return True if the ego vehicle is affected by a traffic light, otherwise return False

    
    def stop_signs_manager(self, waypoint):
        """
        This method is in charge of behaviors for stop signs.
            :param waypoint: the waypoint of the stop sign
            :return affected: True and the stop sign if the ego vehicle is affected by a stop sign, otherwise return False and 0
        """

        stops_list = self._world.get_actors().filter('*stop*') if not self._stops_list else self._stops_list # get all the stop signs in the world
        def dist(v): return v.get_location().distance(waypoint.transform.location) # distance between the waypoint of the stop sign and the ego vehicle
        stops_list = [v for v in stops_list if dist(v) < 20] # filter stop signs within 15 meters from the ego vehicle
        print(str(len(stops_list)) + '\n' if len(stops_list) > 0 else '', end='')
        if len(stops_list) > 1: # if there are more than one stop signs, we sort them by distance
            stops_list.sort(key=dist)
        
        return self._affected_by_stop_sign(self._vehicle, stops_list)[0] # return True if the ego vehicle is affected by a stop sign, otherwise return False

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.
            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
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
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location, right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location, left_wpt.transform.location)

    def _lane_invasion(self, other_vehicle):
        """
        This method is in charge of behaviors for lane invasion.
            :param other_vehicle: other vehicle
            :return lane_invasion: True if the ego vehicle is invading the lane of the other vehicle, otherwise return False
            :return offset: the distance between the waypoint of the other vehicle and effective location of the other vehicle minus the free space on one side of the lane
        """
        other_loc = other_vehicle.get_location() # get the location of the other vehicle
        other_lane_wp = self._map.get_waypoint(other_loc) # get the waypoint of the other vehicle

        other_offset = other_lane_wp.transform.location.distance(other_vehicle.get_location()) # distance between the waypoint of the other vehicle and effective location of the other vehicle
        other_extent = other_vehicle.bounding_box.extent.y # get the extent of the other vehicle
        lane_width = other_lane_wp.lane_width # get the width of the lane
        free_space_on_one_side = lane_width / 2 - other_extent # free space on one side of the lane

        if other_offset > free_space_on_one_side: # if the distance between the waypoint of the other vehicle and effective location of the other vehicle is larger than the free space on one side of the lane
            print('other_offset is larger thant free space on one side')
            return True, other_offset - free_space_on_one_side # return True and the distance between the waypoint of the other vehicle and effective location of the other vehicle minus the free space on one side of the lane
        return False, 0

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*") # get all the vehicles in the world
        def dist(v): return v.get_location().distance(waypoint.transform.location) # distance between the waypoint and the location of the vehicle
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id] # get all the vehicles in the world that are within 45 meters from the waypoint and are not the ego vehicle
        vehicle_list.sort(key=dist) # sort the list of vehicles by distance

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1) # check if there is a vehicle in the left lane
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1) # check if there is a vehicle in the right lane
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60) # check if there is a vehicle in the current lane

            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self._speed > 10 and self._behavior.tailgate_counter == 0: # if there is no vehicle in the current lane and the ego vehicle is not in a junction and the ego vehicle is moving at a speed larger than 10 and the tailgate counter is 0
                self._tailgating(waypoint, vehicle_list)
        print("vehicle_state: ", vehicle_state, "vehicle: ", vehicle, "distance: ", distance)

        return vehicle_state, vehicle, distance # return True if there is a vehicle nearby, False if not, the nearby vehicle, and the distance to the nearby vehicle

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.
            :param waypoint: current waypoint of the agent
            :return walker_state: True if there is a walker nearby, False if not
            :return walker: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*") # get all the walkers in the world
        def dist(w): return w.get_location().distance(waypoint.transform.location) # distance between the waypoint and the location of the walker
        walker_list = [w for w in walker_list if dist(w) < 20] # get all the walkers in the world that are within 10 meters from the waypoint
        walker_list.sort(key=dist) # sort the list of walkers by distance

        if walker_list == []: # if there are no walkers in the world
            return False, None, None

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1) # check if there is a walker in the left lane
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1) # check if there is a walker in the right lane
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=60)  # check if there is a walker in the current lane

        print("walker_state: ", walker_state, "walker: ", walker, "distance: ", distance)
        return walker_state, walker, distance

    def obstacle_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.
            :param waypoint: current waypoint of the agent
            :return obstacle_state: True if there is a object nearby, False if not
            :return obstacle: nearby obstacle
            :return distance: distance to nearby obstacle
        """

        obstacle_list = self._world.get_actors().filter("*static*")  # get all the obstacles in the world
        def dist(w): return w.get_location().distance(waypoint.transform.location) # distance between the waypoint and the location of the obstacle
        obstacle_list = [w for w in obstacle_list if dist(w) < 45]  # get all the obstacles in the world that are within 45 meters from the waypoint
        obstacle_list.sort(key=dist) # sort the list of obstacles by distance

        if obstacle_list == []: # if there are no obstacles in the world
            return False, None, None

        if self._direction == RoadOption.CHANGELANELEFT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1) # check if there is a obstacle in the left lane
        elif self._direction == RoadOption.CHANGELANERIGHT:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1) # check if there is a obstacle in the right lane
        else:
            obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(obstacle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=40)  # check if there is a obstacle in the current lane

        print("obstacle_state", obstacle_state, "obstacle", obstacle, "distance", distance)
        return obstacle_state, obstacle, distance # return True if there is a obstacle nearby, False if not, the nearby obstacle, and the distance to the nearby obstacle

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.
            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        vehicle_speed = get_speed(vehicle) # get the speed of the vehicle in front of us
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6) # difference between our speed and the speed of the vehicle in front of us
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.) # time to collision

        print("VEICOLO DAVANTI. Distance: ", distance, "Velocità ego: ",self._speed, "Velocità veicolo davanti: ", vehicle_speed)
        if self._behavior.safety_time * 0.3 > ttc > 0.0: # if the time to collision is less than the safety time
            target_speed = min([positive(vehicle_speed - self._behavior.speed_decrease), self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) # decrease the speed
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner

        elif self._behavior.safety_time*0.5 > ttc >= self._behavior.safety_time * 0.3: # if the time to collision is between the safety time and twice the safety time
            target_speed = min([max(self._min_speed, vehicle_speed), self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) # keep the speed
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner

        else:
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) # increase the speed
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner

        return control

    def run_step(self, debug=True):  # il debug era false
        """
        Execute one step of navigation.
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information() # update the information about the vehicle

        control = None
        
        # Counters that control the timing of the overtaking, tailgating and stop at stop sign 
        if self._behavior.tailgate_counter > 0: 
            self._behavior.tailgate_counter -= 1

        if self._behavior.overtake_counter > 0:
            self._behavior.overtake_counter -= 1
        
        if self._stay_at_stop_counter > 0:
            self._stay_at_stop_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location() # get the location of the ego vehicle
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc) # get the waypoint of the ego vehicle

        # 1: Traffic lights 
        if self.traffic_light_manager(): 
            # if the ego vehicle is affected by a red light
            return self.emergency_stop()
        
        # 1.1: Stop Signs
        if self.stop_signs_manager(ego_vehicle_wp) and not get_speed(self._vehicle) < 1.0:
                # if the ego vehicle is affected by a stop sign
                self._stay_at_stop_counter=15
                #return self.emergency_stop()
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]): 
            # if the ego vehicle is in a junction and the incoming direction is left or right
            target_speed = min([self._behavior.max_speed, self._speed_limit-5]) # decrease the speed, setting the target speed to the minimum between the maximum speed and the speed limit minus 5
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner
        
        if self._stay_at_stop_counter > 0: 
            #when the counter is still active, the vehicle will not move
            return self.emergency_stop()
        
        # 2.0: Lane Invasion of other vehicles
        vehicle_state_invasion, vehicle_invasion = self._other_lane_occupied_lane_invasion(distance=30) # check if there is a vehicle in our lane
        if vehicle_state_invasion and not self._overtaking_vehicle and not self._overtaking_obj:
            # if there is a vehicle in our lane and we are not overtaking another vehicle or an object we can manage the lane invasion
            invasion_state, offset_invasion = self._lane_invasion(vehicle_invasion)
            if invasion_state:
                # if the lane invasion is true we do a trajectory shift to avoid the vehicle in our lane and we decrease the speed
                #print('LANE INVASION: TRUE, SO DO EMERGENCY STOP')
                self._local_planner.set_lat_offset(-(offset_invasion+0.8)) # set the lateral offset
                self._shrinkage = True
                target_speed = min([self._behavior.max_speed, self._speed_limit]) - (self._behavior.speed_decrease * 2) # decrease the speed
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
        elif self._shrinkage:
            # the normal behavior is restored
            #print('LANE INVASION: FALSE')
            self._local_planner.set_lat_offset(0.0)
            self._shrinkage = False

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars, we use bounding boxes to calculate the actual distance
            distance = w_distance - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            elif distance < 15:
                return self.emergency_stop()

        # 2.2: Obstacle avoidance behaviors
        obstacle_state, obstacle, distance = self.obstacle_avoid_manager(ego_vehicle_wp) # check if there is an obstacle in our lane

        if obstacle_state: 
            # if there is an obstacle in our lane
            distance = distance - max(obstacle.bounding_box.extent.y, obstacle.bounding_box.extent.x) - max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x) # compute the distance between the ego vehicle and the obstacle

            # Emergency brake if the car is very close.
            if self._speed < 0.01: 
                # if the speed of the ego vehicle is less than 0.01
                if ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken: 
                    # check the type of the left lane marking (if it is broken or solid broken)
                    if not self._overtaking_obj and self._direction == RoadOption.LANEFOLLOW: 
                        # if we are not overtaking another vehicle or an object and the direction is lanefollow
                        if not self._other_lane_occupied(distance=self._distance_to_overtake_obj): 
                            # check if there is a vehicle in the left lane
                            self._waypoints_queue_copy = self._local_planner._waypoints_queue.copy() # copy the waypoints queue
                            if self.lane_change("left", self._vehicle_heading, 0, 2, 1.5): 
                                # if we can change lane to the left
                                self._overtaking_obj = True 
                                self._distance_to_overtake_obj = 80 
                                target_speed = max([self._behavior.max_speed, self._speed_limit]) # increase the speed target
                                self._local_planner.set_speed(target_speed) # set the speed
                                control = self._local_planner.run_step(debug=debug) # run the local planner
                                return control
            elif distance < self._behavior.braking_distance and self._speed > 0.01 and not self._overtaking_obj: 
                # if the distance is less than the braking distance and the speed is greater than 0.01 and we are not overtaking another vehicle or an object
                return self.emergency_stop()
            elif distance < 13 and self._speed > 0.01 and not self._overtaking_obj:
                # if the distance is less than 13 and the speed is greater than 0.01 and we are not overtaking another vehicle or an object
                return self.soft_stop()
            elif distance < 30 and self._speed > 0.01 and not self._overtaking_obj:
                # if the distance is less than 30 and the speed is greater than 0.01 and we are not overtaking another vehicle or an object
                return self.no_throttle()
            
            if 40 <= self._distance_to_overtake_obj <= 80:
                # this value is used to check if the vehicle is in the overtaking phase, this value is decrease every time by 0,3, the vehicle moves until it do a overtaking
                self._distance_to_overtake_obj -= 0.3
                

        # 2.x.x: overtake behavior
        if self._ending_overtake: 
            # if we are ending the overtaking phase
            if not self._local_planner.has_incoming_waypoint(): 
                # if the local planner has no incoming waypoint
                self._ending_overtake = False
                self._overtaking_vehicle = False
                self._overtaking_obj = False
                def first_element(t):
                    return t[0]
                route_trace_p = list(map(first_element, self._waypoints_queue_copy))
                route_trace = []
                for i in range ((self._global_planner._find_closest_in_list(ego_vehicle_wp, route_trace_p) ,self._direction)[0], len(self._waypoints_queue_copy)):
                    # for each waypoint in the waypoints queue copy
                    route_trace.append(self._waypoints_queue_copy[i]) # append the waypoint to the route trace
                self._local_planner.set_global_plan(route_trace, True) # set the global plan
                self._behavior.overtake_counter = 50 # set the overtaking counter to 50
            target_speed = min([self._behavior.max_speed, self._speed_limit]) # increase the speed target
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner
            return control
        elif self._overtaking_vehicle or self._overtaking_obj:
            # if we are overtaking another vehicle or an object
            if not self._local_planner.has_incoming_waypoint():
                # if the local planner has no incoming waypoint
                if not self._other_lane_occupied(self._d_max , check_behind=True):
                    # if the other lane is not occupied
                    if self._n_vehicle == 1:
                        # if the number of vehicles in the left lane is 1
                        if self.lane_change("left", self._vehicle_heading, 0, 2, 1.2):
                            # normal overtaking end, it used when there is only one vehicle in the left lane
                            self._ending_overtake = True
                            self._n_vehicle == 0
                    else:
                        if self.lane_change("left", self._vehicle_heading, 0, 1.85, 0.6):
                            # aggressive overtaking end, it used when there are more than 2 vehicles in the left lane
                            self._ending_overtake = True
                            self._n_vehicle == 0
                else:
                    # if the left lane, to do re-entry from overtaking, is occupied we continue to overtake
                    if self._n_vehicle <= 2: 
                        # if the number of vehicles in the left lane is 2 or less
                        self.lane_change("left", self._vehicle_heading, 0.85, 0, 0)
                    else:
                        self.lane_change("left", self._vehicle_heading, 0.89, 0, 0)

            target_speed = max([self._behavior.max_speed, self._speed_limit]) # increase the speed target
            self._local_planner.set_speed(target_speed) # set the speed
            control = self._local_planner.run_step(debug=debug) # run the local planner
            return control

        # 2.3: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp) # check if there is a vehicle in our lane

        if vehicle_state: 
            # if there is a vehicle in our lane
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x) # Distance is computed from the center of the two cars, we use bounding boxes to calculate the actual distance

            if (ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken) and self._behavior.overtake_counter == 0 and distance < 6: 
                # check the type of the left lane marking (if it is broken or solid broken) and if the overtaking counter is 0 and the distance is less than 6
                if not self._overtaking_vehicle and self._direction == RoadOption.LANEFOLLOW:
                    # if we are not overtaking another vehicle and the direction is lanefollow
                    if self._is_slow(vehicle): 
                        # if the vehicle in front of us is slow
                        stuck, self._n_vehicle, self._distance_to_over, self._d_max  = self._iam_stuck(ego_vehicle_wp) # check if we are stuck
                        vehicle_list = self._world.get_actors().filter("*vehicle*") # get all the vehicles in the world
                        def dist(v, w): return v.get_location().distance(w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x # distance between the two vehicles
                        vehicle_list = [v for v in vehicle_list if dist(v, self._vehicle) < 30 and v.id != self._vehicle.id] # get all the vehicles in the world that are within 30 meters from the ego vehicle and are not the ego vehicle

                        new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=180, lane_offset=-1) # check if there is a vehicle in the total of left lane
                        new_vehicle_state2, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit), low_angle_th=90, up_angle_th=180, lane_offset=-1) # check if there is a vehicle in the left lane behind us

                        if not new_vehicle_state and not new_vehicle_state2:  

                            if not self._other_lane_occupied(distance=self._distance_to_over) and not self._overtaking_vehicle and self.closest_intersection() > 100:
                                # if the other lane is not occupied and we are not overtaking another vehicle and we are not in a junction
                                self._waypoints_queue_copy = self._local_planner._waypoints_queue.copy() # copy the waypoints queue
                                if self.lane_change("left", self._vehicle_heading, 0, 2, 1.5): 
                                    # if we can change lane to the left
                                    self._overtaking_vehicle = True
                                    target_speed = max([self._behavior.max_speed, self._speed_limit]) # increase the speed target
                                    self._local_planner.set_speed(target_speed) # set the speed
                                    control = self._local_planner.run_step(debug=debug) # run the local planner
                                    return control

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([self._behavior.max_speed, self._speed_limit-5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 3.1: Normal behavior - controllo se la velocità è troppo alta mentre sto sterzando
        elif self._speed > 45 and self._steer > 90 or self.closest_intersection() < 30:
            # if the speed is too high while steering or we are in the proximity of an intersection we decelerate
            return self.decelerate()

        # 4: Normal behavior
        else:
            # if there are no pedestrians, no vehicles in front of us, we proceed normally
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

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

    def decelerate(self):
        """
        This method is in charge of decelerating the vehicle when there's a turn ahead.
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()
        self._local_planner.set_speed(30)
        control = self._local_planner.run_step()
        return control

    def soft_stop(self):
        """
        This method is in charge of braking the vehicle
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.2
        control.hand_brake = False
        return control

    def no_throttle(self):
        """
        This method is in charge of not accelerating the vehicle
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        return control

    def closest_intersection(self):
        """
            This method is in charge of finding the closest intersection.
            :return closest_distance: the distance to the closest intersection
        """
        intersections = [] # list of all the intersections
        # get all the waypoints in the queue that are intersections
        for i in range(len(self._local_planner._waypoints_queue)):
            if self._local_planner._waypoints_queue[i][0].is_junction:
                intersections.append(self._local_planner._waypoints_queue[i][0])

        vehicle_location = self._vehicle.get_location() # get the location of the ego vehicle
        vehicle_yaw = math.radians(self._vehicle.get_transform().rotation.yaw) # get the yaw of the ego vehicle
        closest_intersection = None # closest intersection
        closest_distance = float('inf') # distance to the closest intersection
        # find the closest intersection
        for intersection in intersections:
            intersection_location = intersection.transform.location # location of the intersection
            intersection_direction = math.atan2(intersection_location.y - vehicle_location.y, intersection_location.x - vehicle_location.x) # direction of the intersection 
            intersection_distance = math.sqrt((intersection_location.x - vehicle_location.x)**2 + (intersection_location.y - vehicle_location.y)**2) # distance to the intersection
            relative_direction = abs(math.degrees(vehicle_yaw - intersection_direction)) # relative direction between the ego vehicle and the intersection
            # if the relative direction is less than 90 degrees and the distance to the intersection is less than the closest distance
            if relative_direction <= 90 and intersection_distance < closest_distance: 
                closest_intersection = intersection_location # update the closest intersection
                closest_distance = intersection_distance # update the closest distance

        # CANCELLAREEEEE!!!!!!
        if closest_intersection is not None:
            print('Closest intersection:', closest_intersection, 'Distance:', closest_distance)
        else:
            print('No intersections found.')

        return closest_distance
