
class Agent(object):

    def __init__(self, vehicle):

        self._vehicle = vehicle
        self._proximity_tlight_threshold = 5.0
        self._proximity_vehicle_threshold = 10.0
        self._local_planner = None
        self._world = self._vehicle.get_world()
        try:
            self._map = self._world.get_map()

        except RuntimeError as error:

            print('RuntimeError : {}' .format(error))

        self._last_traffic_light = None

    def _bh_is_vehicle_hazard(self, ego_wpt, ego_loc, vehicle_list, proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):

        print('_bh_is_vehicle_hazard called')

        if ego_wpt.lane_id < 0 and lane_offset != 0:

            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()

            target_wpt = self._map.get_waypoint(target_vehicle_loc)

            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:

                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[
                    0]

                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:

                    continue

            if is_within_distance(target_vehicle_loc, ego_loc, self._vehicle.get_transform().rotation.yaw, proximity_th, up_angle_th, low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))

        return (False, None, -1)

    @staticmethod
    def emergency_stop():

        print('emergency_stop called ')

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control
