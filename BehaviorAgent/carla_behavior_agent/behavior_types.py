# Autonomous Vehicle Driving Project.
# Copyright (C) 2023 - All Rights Reserved
# Group:
#   Faiella Ciro              0622701816  c.faiella8@studenti.unisa.it
#   Giannino Pio Roberto      0622701713	p.giannino@studenti.unisa.it
#   Scovotto Luigi            0622701702  l.scovotto1@studenti.unisa.it
#   Tortora Francesco         0622701700  f.tortora21@studenti.unisa.it

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0
    overtake_counter = 0

class Normal(object):
    """Class for Normal agent."""
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10 # 6
    braking_distance = 5
    tailgate_counter = 0
    overtake_counter = 0

class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 4
    tailgate_counter = -1
    overtake_counter = 0
