import torch

from models.game_data import GameData


class ComputeRewards():
    def __init__(self, dtype, device):
        self.previous_dataes = []
        self.dtype = dtype
        self.device = device

    def get_reward(self, data):
        """
        This method will call the getReward and calReward functions to calculate the rewards and return the final reward.
        """
        reward = getReward(data, self.previous_dataes)
        self.previous_dataes.append(data)
        if len(self.previous_dataes) > 5:
            self.previous_dataes.pop(0)
        return torch.tensor(reward, dtype=self.dtype, device=self.device)

def getReward(data, previous_dataes):
    """calculates the reward based on the current frame's data and the previous frames' data.    """

    if len(previous_dataes) > 0:
        health_reward = 0
        if data.car.Health - previous_dataes[0].car.Health < 0:
            health_reward = (data.car.Health -
                             previous_dataes[0].car.Health) * 2
        reward = calReward(data) + health_reward
    else:
        reward = calReward(data)
    return reward

def calReward(data):
    """calculates the reward based on the current frame's data.    """

    assert isinstance(data, GameData)
    time_since = 150  # ms
    against_traffic = time_since > data.time_since_player_drove_against_traffic >= 0
    drove_on_pavement = time_since > data.time_since_player_drove_on_pavement >= 0
    hit_ped = time_since > data.time_since_player_hit_ped >= 0
    hit_vehicle = time_since > data.time_since_player_hit_vehicle >= 0
    num_near_by_vehicle = len(data.near_by_vehicles)
    len_near_by_touching_peds = len(data.near_by_touching_peds)
    len_near_by_touching_props = len(data.near_by_touching_props)
    len_near_by_touching_vehicles = len(data.near_by_touching_vehicles)
    diff_Speed = abs(data.car.Speed - 30)
    is_player_in_water = data.is_player_in_water
    reward = 0
    if data.onRoad:
        reward += 1
    if not against_traffic:
        reward += 1
    if not drove_on_pavement:
        reward += 1
    if not hit_ped:
        reward += 1
    if not hit_vehicle:
        reward += 1
    reward -= diff_Speed
    if not is_player_in_water:
        reward += 1
    reward -= num_near_by_vehicle
    reward -= len_near_by_touching_peds
    reward -= len_near_by_touching_props
    reward -= len_near_by_touching_vehicles
    return reward