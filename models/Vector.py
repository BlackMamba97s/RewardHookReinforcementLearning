class Vector:
    """A vector
    """

    def __init__(self, vector=None):
        """Retrieve the values of X Y Z keys from a dictionary"""

        if vector is None:
            self.X = None
            self.Y = None
            self.Z = None
        else:
            self.X = vector['X']
            self.Y = vector['Y']
            self.Z = vector['Z']