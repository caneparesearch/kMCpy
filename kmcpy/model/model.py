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
    def compute(self, *args, **kwargs):
        """
        Generic computation method for the model. This method must be implemented by subclasses.
        Can return either site energy or barrier energy depending on the model configuration.
        
        Returns:
            float: The computed energy value (site energy, barrier energy, etc.)
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def compute_probability(self, *args, **kwargs):
        """
        Compute the transition probability based on the model's parameters and the current state.
        This method must be implemented by subclasses.
        
        Returns:
            float: The computed transition probability.
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


class CompositeModel(BaseModel):
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    def __str__(self):
        return f"CompositeModel with {len(self.models)} models"

    def __repr__(self):
        return f"CompositeModel(models={self.models}, weights={self.weights})"
    
    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Compute the transition probability based on the model's parameters and the current state.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def compute_probability(self, *args, **kwargs):
        """
        Compute the transition probability based on the model's parameters and the current state.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def as_dict(self):
        """
        Convert the composite model object to a dictionary representation.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def from_dict(cls, d):
        """
        Create a composite model object from a dictionary representation.
        This method must be implemented by subclasses.
        
        Args:
            d (dict): Dictionary representation of the model.
        
        Returns:
            CompositeModel: An instance of the composite model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
