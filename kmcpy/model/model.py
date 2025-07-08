from abc import ABC, abstractmethod
import json
import logging
from kmcpy.io import convert

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class BaseModel(ABC):
    """
    Base class for models in kmcpy. This class is intended to be inherited by other model classes.
    It provides a common interface for serialization and deserialization of model objects.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseModel. This method can be overridden by subclasses to handle specific initialization.
        """
        self.name = kwargs.get("name", None)

    @abstractmethod
    def __str__(self):
        """
        Return a string representation of the model object.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def __repr__(self):
        """
        Return a detailed string representation of the model object.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")    
    
    @abstractmethod
    def compute_probability(self, *args, **kwargs):
        """
        Compute probability based on the model. This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def as_dict(self):
        """
        Convert the model object to a dictionary representation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_json(cls, fname):
        """
        Load a model object from a JSON file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def to_json(self, fname):
        """
        Save the model object to a JSON file.
        """
        logger.info("Saving model to: %s", fname)
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )
            fhandle.write(jsonStr)
