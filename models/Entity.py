from models.Vector import Vector


class Entity:

    def __init__(self, entity):
        # Create an Entity object from a dictionary by getting the corresponding keys values

        self.IsOnScreen = entity['IsOnScreen']  # Whether the object is on the screen
        self.Position = Vector(entity['Position'])  # The position
        self.RightVector = Vector(entity['RightVector'])  # The right direction vector
        self.Rotation = Vector(entity['Rotation'])  # The rotation direction
        self.ForwardVector = Vector(entity['ForwardVector'])  # The forward direction vector
        self.HeightAboveGround = entity['HeightAboveGround']  # The distance above ground
        self.Model = entity['Model']  # The kind of object
        self.Velocity = Vector(entity['Velocity'])  # The velocity vector
