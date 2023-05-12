# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance,
                               distance_vehicle)
# from perception.perfectTracker.gt_tracker import PerfectTracker

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        
        self._last_stop_sign = None
        self._last_traffic_light = None

        self._junction_counter = 0

        # Base parameters
        self._ignore_traffic_lights = True  # CHANGE TO FALSEEE!!!!
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

        self._stops_list = self._world.get_actors().filter("*stop*")
        self._stops_map = {} # Dictionary mapping a stop sing to a wp corrspoing to its trigger volume location
 
    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def _vehicle_next_junction(self, waypoint, vehicle_list=None, check_lane='left'):
        if self._ignore_vehicles:
            return (False, None, -1)

        if vehicle_list is None:
            vehicle_list = self._world.get_actors().filter("*vehicle*")
            def dist(v): return v.get_location().distance(waypoint.transform.location)
            vehicle_list = [v for v in vehicle_list if dist(v) < 43 and v.id != self._vehicle.id]
            vehicle_list.sort(key=dist)

        junction = waypoint.get_junction()

        def _print_vehicle_info(vehicle, is_ego=False):
            print('||||| vehicle_type:' if not is_ego else '||||| ego:', end='')
            print(vehicle.type_id, ' ||||| data: transform', vehicle.get_transform(), ' forward', vehicle.get_forward_vector(), ' right', vehicle.get_right_vector())

        for vehicle in vehicle_list:
            print('--------------- for ----------------')
            ve_wpt = self._map.get_waypoint(vehicle.get_location())
            if junction is not None and ve_wpt.get_junction().id == junction.id:
                _print_vehicle_info(vehicle)
                _print_vehicle_info(self._vehicle, True)
                # print('||||| vehicle_type:', vehicle.type_id, ' ||||| data: transform', vehicle.get_transform(), ' forward', vehicle.get_forward_vector(), ' right', vehicle.get_right_vector())
                # print('||||| ego:', self._vehicle.type_id, ' ||||| data: transform', self._vehicle.get_transform(), ' forward', self._vehicle.get_forward_vector(), ' right', self._vehicle.get_right_vector())
        '''
        if check_lane != 'left' or check_lane != 'right':
            return (False, None, -1)
        elif check_lane == 'left':
            return self._vehicle_obstacle_detected(vehicle_list, up_angle_th=90, lane_offset=-1)
        elif check_lane == 'right':
            return self._vehicle_obstacle_detected(vehicle_list, up_angle_th=90, lane_offset=1)
        else:
            return (False, None, -1)
        '''

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution # prova xon 1.5
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None) 

        if lights_list is None:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if max_distance is None:
            max_distance = self._base_tlight_threshold 

        if self._last_traffic_light:
            """se ci siamo già fermati al sem, o ci stiamo per fermare"""
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        # valuto tutti gli altri semafori
        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id: # if it's on the same road of the ego
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector() # valuriamo anceh l'orientamento del veicolo per non cacare i sem che stanno dietro
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z # dove dbb andare 

            if dot_ve_wp < 0: # sta dietro
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]): # valutazione sulla distanza
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)
    
    def _affected_by_stop_sign(self, vehicle=None, stops_list=None, max_distance=None):
        if self._ignore_stop_signs:
            return (False, None) # False = non cosideriamo il segnale, None = nessun oggetto segnale, -1 = distanza dall'oggetto
        
        if stops_list is None:
            stops_list = self._stops_list
        
        if vehicle is None:
            vehicle = self._vehicle
        
        if max_distance is None:
            max_distance = self._base_vehicle_threshold
        
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if self._last_stop_sign is not None:
            print('-------- last_stop_sign if --------------')
            l_vehicle_state, l_vehicle, l_distance = self._vehicle_next_junction(ego_vehicle_waypoint, check_lane='left')
            r_vehicle_state, r_vehicle, r_distance = self._vehicle_next_junction(ego_vehicle_waypoint, check_lane='right')
            
            if not l_vehicle_state and not r_vehicle_state:
                self._last_stop_sign = None
            elif (l_vehicle_state and not r_vehicle_state) or (l_vehicle_state and r_vehicle_state):
                return (True, self._last_stop_sign)
            elif (r_vehicle_state and not l_vehicle_state):
                # avanaza fino al waypoint prima della collisione
                dist = distance_vehicle(ego_vehicle_waypoint, r_vehicle.get_transform())
                pass
            else:
                return (True, self._last_stop_sign)

        for stop_sing in stops_list:
            if stop_sing.id in self._stops_map:
                trigger_wp = self._stops_map[stop_sing.id]
            else:
                trigger_location = get_trafficlight_trigger_location(stop_sing) # is_within_trigger_volume(vehicle, stop_sing, stop_sing.trigger_volume) # ottengo un boolean
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._stops_map[stop_sing.id] = trigger_wp
            
            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance: continue
                
            if trigger_wp.road_id != ego_vehicle_waypoint.road_id: continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector() # valuriamo anche l'orientamento del veicolo
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z # dove dobbiamo andare

            if dot_ve_wp < 0: continue

            if self._last_stop_sign is not None:
                if trigger_wp == self._stops_map[self._last_stop_sign.id]: continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]): # valutazione sulla distanza
                self._last_stop_sign = stop_sing
                return (True, stop_sing)

        return (False, None)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        ) # informazioni sul mio veicolo

        for target_vehicle in vehicle_list: # per ogni veicolo della lista 
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any) # vedo dove si trova quel veicolo

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction: # se non siamo in un incrocio fa delle cose 

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0] # valuto il veicolo un po' più avanti perchè quello subito prima è troppo vicino 
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                # valuto la posizone e direzione del veicolo 
                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform # prendo il punto posteriore del veicolo
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]): # vedo la distanza tra la mia punta e il culo dell'altro veicolo
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location)) # è una possivile colluisione con quel veiclo 

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                # cotruisco un poligono sulla mia posizione
                route_bb = [] # creo una lista vuota, dove ci saranno i miei punti estremi e poi creo il poligono
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y # metà della distanza ruota destra ruota sinistra totale del veicolo
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                # una volta preso il poligono faccio:
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id: # se sono io stesso non mi interessa
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance: # se sono troppo lontani non mi interessa
                        continue 

                    # se mi crea collisione con un altro veicolo allora mi interessa
                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon): # se si intersecano allora è una possibile collisione e mi da fastidio
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location)) # dato che impatta voglio sapere le info del veicolo che mi ha impattato

                return (False, None, -1)

        return (False, None, -1)

    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=False, lane_changes=1, step_distance=2): #check era true
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return [] #plan
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            #if curr_orientation != side_orientation and direction == "left":
            #    next_wps = plan[-1][0].previous(step_distance)
            #else:
            #    next_wps = plan[-1][0].next(step_distance)

            next_wps = plan[-1][0].next(lane_change_distance) # prima dell'if era così
            if not next_wps:
                return [] #plan
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return [] #plan
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return [] #plan
                side_wp = next_wp.get_right_lane() # provare anche con solo next_wp 

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return [] #plan

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        curr_orientation = next_wp.transform.rotation.yaw
        side_orientation = side_wp.transform.rotation.yaw
        
        # Other lane
        distance = 0
        while distance < distance_other_lane:
            if curr_orientation != side_orientation and direction == "left":
                next_wps = plan[-1][0].previous(step_distance)
            else:
                next_wps = plan[-1][0].next(step_distance)
            
            if not next_wps:
                return [] #plan
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan