# channel = 3
# leftx = 0
# lefty = 20
# small_map_leftx = leftx + 5
# small_map_lefty = lefty + 465
# small_map_pos = [[samll_map_leftx, samll_map_lefty],
#                  [samll_map_leftx + 160, samll_map_lefty + 115]]
# screen_pos = [[leftx, lefty], [leftx + screen_width,
#                                lefty + screen_height]]
num_episodes = 2000 # number of episodes, should be around 10k, but im lowering for now
num_steps = 40 # number of steps for episode, relatively in my car game its the frame itself
near_by_vehicles_limit = 8  # number of nearby vehicles to enter into the network
near_by_peds_limit = 5  # number of nearby pedestrians to enter into the network
near_by_props_limit = 20  # number of nearby objects to enter into the network
near_by_touching_vehicles_limit = 5  # number of nearby touching vehicles to enter into the network
near_by_touching_peds_limit = 5  # number of nearby touching pedestrians to enter into the network
near_by_touching_props_limit = 10  # number of nearby touching objects to enter into the network
action_list = ['w', 'wa', 'wd', 'sa', 'sd', "s"]
key_list = "\bABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\"
stuck_threshold=2
health_threshold=50