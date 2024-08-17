# Example class
class BaseClass:
    """
    Base class representing an entity.

    Attributes
    ----------
    name : str
        The name of the entity.

    Methods
    -------
    __init__():
        Initializes a new instance of the BaseClass.
    __str__():
        Returns a string representation of the entity.
    __repr__():
        Returns a string representation of the entity for debugging.
    __eq__(other):
        Checks if two entities are equal based on their names.

    """

    def __init__(self, name: str):
        """
        Initializes a new instance of the BaseClass.
        """
        self.name = name

    def __str__(self):
        """
        Returns a string representation of the entity.
        """
        return self.name

    def __repr__(self):
        """
        Returns a string representation of the entity for debugging.
        """
        return self.name

    def __eq__(self, other):
        """
        Checks if two entities are equal based on their names.

        Parameters
        ----------
        other : BaseClass
            Another instance of BaseClass.

        Returns
        -------
        bool
            True if the entities are equal, False otherwise.
        """
        return self.name == other.name

    def something(self):
        """
        Does something.
        """
        return "something"
