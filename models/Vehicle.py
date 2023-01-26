from models.Vector import Vector


class Vehicle:
    def __init__(self, vehicle):
        self.IsOnScreen = vehicle['IsOnScreen']
        self.Position = Vector(vehicle['Position'])
        self.RightVector = Vector(vehicle['RightVector'])
        self.Rotation = Vector(vehicle['Rotation'])
        self.ForwardVector = Vector(vehicle['ForwardVector'])
        self.HeightAboveGround = vehicle['HeightAboveGround']
        self.Velocity = Vector(vehicle['Velocity'])
        self.RightHeadLightBroken = vehicle['RightHeadLightBroken']
        self.LeftHeadLightBroken = vehicle['LeftHeadLightBroken']
        self.LightsOn = vehicle['LightsOn']
        self.EngineRunning = vehicle['EngineRunning']
        self.Health = vehicle['Health']
        self.MaxHealth = vehicle['MaxHealth']
        self.SearchLightOn = vehicle['SearchLightOn']
        self.IsOnAllWheels = vehicle['IsOnAllWheels']
        self.IsStoppedAtTrafficLights = vehicle['IsStoppedAtTrafficLights']
        self.IsStopped = vehicle['IsStopped']
        self.IsDriveable = vehicle['IsDriveable']
        self.IsConvertible = vehicle['IsConvertible']
        self.IsFrontBumperBrokenOff = vehicle['IsFrontBumperBrokenOff']
        self.IsRearBumperBrokenOff = vehicle['IsRearBumperBrokenOff']
        self.IsDamaged = vehicle['IsDamaged']
        self.Speed = vehicle['Speed']
        self.BodyHealth = vehicle['BodyHealth']
        self.MaxBraking = vehicle['MaxBraking']
        self.MaxTraction = vehicle['MaxTraction']
        self.EngineHealth = vehicle['EngineHealth']
        self.SteeringScale = vehicle['SteeringScale']
        self.SteeringAngle = vehicle['SteeringAngle']
        self.WheelSpeed = vehicle['WheelSpeed']
        self.Acceleration = vehicle['Acceleration']
        self.FuelLevel = vehicle['FuelLevel']
        self.CurrentRPM = vehicle['CurrentRPM']
        self.CurrentGear = vehicle['CurrentGear']
        self.HighGear = vehicle['HighGear']