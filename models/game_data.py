from models.Entity import Entity
from models.Player import Player
from models.Vector import Vector
from models.Vehicle import Vehicle


class GameData:
    def __init__(self, data):
        self.charactor = Player(data['charactor'])  # The player
        self.car = Vehicle(data['car'])  # The vehicle
        self.endPosition = Vector(data['endPosition'])  # The end position
        self.startPosition = Vector(data['startPosition'])  # The starting position
        self.time_since_player_drove_against_traffic = data[
            'time_since_player_drove_against_traffic']  # The time since the player last drove against traffic
        self.time_since_player_drove_on_pavement = data[
            'time_since_player_drove_on_pavement']  # The time since the player last drove on the pavement
        self.time_since_player_hit_ped = data[
            'time_since_player_hit_ped']  # The time since the player last hit a pedestrian
        self.time_since_player_hit_vehicle = data[
            'time_since_player_hit_vehicle']  # The time since the player last hit a vehicle
        self.near_by_vehicles = [Vehicle(item)
                                 for item in data['near_by_vehicles']]  # Nearby vehicles
        self.near_by_peds = [Player(item)
                             for item in data['near_by_peds']]  # Nearby pedestrians
        self.near_by_props = [Entity(item)
                              for item in data['near_by_props']]  # Nearby objects (not including map objects)
        self.near_by_touching_peds = [
            Player(item) for item in data['near_by_touching_peds']]  # Nearby touching vehicles
        self.near_by_touching_vehicles = [
            Vehicle(item) for item in data['near_by_touching_vehicles']]  # Nearby touching pedestrians
        self.near_by_touching_props = [
            Entity(item) for item in data['near_by_touching_props']]  # Nearby touching objects
        self.next_position_on_street = Vector(
            data['next_position_on_street'])  # The next position on the street
        self.forward_vector3 = Vector(data['forward_vector3'])  # The forward direction of the vehicle
        self.radius = data['radius']  # The nearby range (radius)
        self.onRoad = data['onRoad']  # Whether on the road (not including sidewalks and greenery)
        self.is_ped_injured = data['is_ped_injured']  # Whether the player is injured
        self.is_ped_in_any_vehicle = data['is_ped_in_any_vehicle']  # Whether the player is in a vehicle
        self.is_player_in_water = data['is_player_in_water']  # Whether the vehicle is in water
