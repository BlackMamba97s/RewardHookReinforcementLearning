from models.Vector import Vector


class Player:

    def __init__(self, player):
        self.IsHuman = player['IsHuman']  # Whether it's human
        self.IsOnFoot = player['IsOnFoot']  # Whether it's standing
        self.IsPlayer = player['IsPlayer']  # Whether it's the player itself
        self.IsOnScreen = player['IsOnScreen']  # Whether it's on the screen
        self.Position = Vector(player['Position'])  # The position
        self.RightVector = Vector(player['RightVector'])  # The right direction vector
        self.Rotation = Vector(player['Rotation'])  # The rotation direction
        self.ForwardVector = Vector(player['ForwardVector'])  # The forward direction vector
        self.Velocity = Vector(player['Velocity'])  # The velocity vector
