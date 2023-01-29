import json
import time

import numpy as np
import requests
import torch

from models.game_data import GameData
from util.utils import minus, vector_to_numpy, vectors_to_numpy, near_by_vectors, distance, calculate_xy_angle
from constants import *


def getAssistInfo(data):
    assist = []
    speed = data.car.Speed
    acceleration = data.car.Acceleration
    target_direction = minus(data.car.Position, data.next_position_on_street)
    distanceToEndPosition = distance(data.car.Position, data.endPosition)
    angle = calculate_xy_angle(target_direction, data.forward_vector3)
    endPositionVector = vector_to_numpy(minus(data.car.Position, data.endPosition))
    near_by_vehicles = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_vehicles], near_by_vehicles_limit))
    near_by_peds = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_peds if item.IsOnFoot], near_by_peds_limit))
    near_by_props = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_props], near_by_props_limit))
    near_by_touching_vehicles = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_touching_vehicles], near_by_touching_vehicles_limit))
    near_by_touching_peds = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_touching_peds], near_by_touching_peds_limit))
    near_by_touching_props = vectors_to_numpy(near_by_vectors(data.car.Position, [item.Position for item in data.near_by_touching_props], near_by_touching_props_limit))
    assist.append(speed) # current speed
    assist.append(angle) # angle between speed and target direction
    assist.append(acceleration) # acceleration (0, 1, -1)
    assist.append(data.car.SteeringScale) # steering
    assist.append(data.car.SteeringAngle) # steering angle
    assist.append(data.car.CurrentRPM) # current RPM
    assist.append(1 if data.car.IsOnScreen else 0) # if the car is on screen
    assist.append(1 if data.onRoad else 0) # if the car is on the road (not on a sidewalk or median)
    assist.append(distanceToEndPosition) # distance to the end position
    assist.extend(vector_to_numpy(data.car.Velocity)) # velocity vector
    assist.extend(vector_to_numpy(data.car.ForwardVector)) # forward direction vector
    assist.extend(vector_to_numpy(data.car.RightVector)) # right direction vector
    assist.extend(vector_to_numpy(data.car.Rotation)) # car rotation angle
    assist.extend(endPositionVector.flatten()) # direction vector to the end position (from current position)
    assist.extend(near_by_vehicles.flatten()) # direction vectors to nearby vehicles (from current position)
    assist.extend(near_by_peds.flatten()) # direction vectors to nearby pedestrians (from current position)
    assist.extend(near_by_props.flatten()) # direction vectors to nearby props (from current position)
    assist.extend(near_by_touching_vehicles.flatten()) # direction vectors to nearby touching vehicles (from current position)
    assist.extend(near_by_touching_peds.flatten()) # direction vectors to nearby touching pedestrians (from current position)
    assist.extend(near_by_touching_props.flatten()) # direction vectors to nearby touching props (from current position)
    assist = np.array(assist)
    assist = np.reshape(assist, newshape=(1, len(assist)))
    return assist


def get_game_data(fromNet):
    if fromNet:
        url = "http://localhost:31730/data"
        response = requests.get(url)
        return json.loads(response.content)
    else:
        with open("./util/sample.json") as f:
            data = json.load(f)
            return data


def get_input_data(dtype, device):
    while True:
        try:
            print("sono qua")
            json = get_game_data(fromNet=True)
            print(json)
            data = GameData(json)
            info = getAssistInfo(data)
            info = torch.tensor(info, dtype=dtype, device=device)
            return data, info
        except Exception as e:
            print(e)