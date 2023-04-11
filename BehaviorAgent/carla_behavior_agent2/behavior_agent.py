
class BehavoirAgent(Agent):

    def __init__(self, vehicle, ignore_traffic_light=False, behavior='normal'):

        super(BehavoirAgent, self).__init__(vehicle)
        self.vehicle = vehicle
        self.ignore_traffic_light = ignore_traffic_light
        self._local_planner = LocalPlanner(self)
        self._grp = None
        self.look_ahead_steps = 0

        # vehicle information

        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.light_id_to_ignore = -1
        self.min_speed = 5
        self.behavior = None
        self._sampling_resolution = 4.5

        if behavior == 'normal':

            self.behavior = Normal()

    def update_information(self):

        print('update_information called')

        self.speed = get_speed(self.vehicle)
        self.speed_limit = self.vehicle.get_speed_limit()

        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option

        if self.direction is None:

            self.direction = RoadOption.LANEFOLLOW

        self.look_ahead_steps = int((self.speed_limit)/10)

        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)

        if self.incoming_direction is None:

            self.direction = RoadOption.LANEFOLLOW

        self.is_at_traffic_light = self.vehicle.is_at_traffic_light()

        if self.ignore_traffic_light:

            self.light_state = "Green"

        else:

            self.light_state = str(self.vehicle.get_traffic_light_state())

    def set_destination(self, start_location, end_location, clean=False):

        print('set_destination called')

        if clean:

            self._local_planner.waypoints_queue.clear()

        self.start_waypoint = self._map.get_waypoint(start_location)

        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        self._local_planner.set_global_plan(route_trace, clean)

    def _trace_route(self, start_waypoint, end_waypoint):

        print('_trace_route BehavoirAgent called')

        if self._grp is None:
            wld = self.vehicle.get_world()

            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)

            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        route = self._grp.trace_route(
            start_waypoint.transform.location, end_waypoint.transform.location)

        return route

    def reroute(self, spawn_points):

        print('reroute called ')

        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location

        self.set_destination(new_start, destination)

    def traffic_light_manager(self, waypoint):

        print('traffic_light_manager called')

        light_id = self.vehicle.get_traffic_light(
        ).id if self.vehicle.get_traffic_light() is not None else -1

        if self.light_state == "Red":

            if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):

                return 1

            elif waypoint.is_junction and light_id != -1:

                self.light_id_to_ignore = light_id

        if self.light_id_to_ignore != light_id:

            light_id_to_ignore = -1

        return 0

    def _overtake(self, location, waypoint, vehicle_list):

        print('overtake called 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

        left_turn = None
        right_turn = None

        left_wpt = waypoint.get_left_lane()

        right_wpt = waypoint.get_right_lane()

        if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:

            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1)

            if not new_vehicle_state:

                self.behavior.overtake_counter = 200
                self.set_destination(
                    left_wpt.transform.location, self.end_waypoint, clean=True)

        elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:

            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1)

            if not new_vehicle_state:

                self.behavior.overtake_counter = 200
                self.set_destination(
                    right_wpt.transform.location, self.end_waypoint.transform.location, clean=True)

    def _tailgating(self, location, waypoint, vehicle_list):

        print('tailgating called ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt')

        left_turn = waypoint.left_lane_marking.lane_change

        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
            self.behavior.min_proximity_threshold, self.speed_limit/2), up_angle_th=180, low_angle_th=160)

          if behind_vehicle_state and self.speed < get_speed(behind_vehicle):

               if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:

                    new_vehicle, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                        self.behavior.min_proximity_threshold, self.speed_limit/2), up_angle_th=180, lane_offset=1)

                      if not new_vehicle:

                           print('tailgating moving towards right')
                            self.behavior.tailgate_counter = 200
                            self.set_destination(
                                right_wpt.transform.location, end_waypoint.transform.location, clean=True)

                elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:

                    new_vehicle, _, _ = self._bh_is_vehicle_hazard(waypoint, location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset = -1)

                    if not new_vehicle_state:

                        print('tailgating , moving to the left !')
                        self.behavior.tailgate_counter = 200
                        self.set_destination(left_wpt.transform.location, self.end_waypoint.transform.location, clean= True)

    def collision_and_car_avoid_manager(self, location, waypoint):

        print('collision_and_car_avoid_manager called')

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):

            return v.get_location().distance(waypoint.transform.location)

        vehicle_list = [v for v in vehicle_list if dist(
            v) < 45 and v.id != self.vehicle.id]

        if self.direction == RoadOption.CHANGELANELEFT:

            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset =-1)

        elif self.direction == RoadOption.CHANGELANERIGHT:

            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset = 1)

        else:

            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/3 ) , up_angle_th = 30 )

            if vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10 and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):

                self._overtake(location, waypoint, vehicle_list)

            elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10 and self.behavior.tailgate_counter == 0:

                self._tailgating(location, waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def car_following_manager(self, vehicle, distance, debug = False) :

        print('car car_following_manager called')

        vehicle_speed = get_speed(vehicle)

        delta_v = max(1, (self.speed - vehicle_speed)/3.6)

        ttc = distance/delta_v if delta_v != 0 else distance / \
            np.nextafter(0., 1.)

        if self.behavior.safety_time > ttc > 0.0:

            control = self._local_planner.run_step(target_speed = min(positive(vehicle_speed - self.behavior.speed_decrease), min (self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug = debug)

        elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time:

            control = self._local_planner.run_step(target_speed = min(max(self.min_speed, vehicle_speed), min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)) , debug = debug)

        else:

            control = self._local_planner.run_step(target_speed = min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug = debug)

        return control

    def run_step(self, debug = False):

        print('run_step function called')

        self.update_information()

        control = None
        if self.behavior.tailgate_counter > 0:

            self.behavior.tailgate_counter -= 1

        if self.behavior.overtake_counter > 0:

            self.behavior.overtake_counter -= 1

        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        if self.traffic_light_manager(ego_vehicle_wp) != 0:

            return self.emergency_stop()

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_loc, ego_vehicle_wp)

        if vehicle_state:

            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

            if distance < self.behavior.braking_distance:

                return self.emergency_stop()

            else:

                control = self.car_following_manager(vehicle, distance)

        elif self.incoming_waypoint and self.incoming_waypoint.is_junction and (self.incoming_direction == RoadOption.LEFT or self.incoming_direction == RoadOption.RIGHT):

            control = self._local_planner.run_step(target_speed= min(self.behavior.max_speed, self.speed_limit - 5), debug = debug)

        else:

            control = self._local_planner.run_step(target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug = debug)

        return control
