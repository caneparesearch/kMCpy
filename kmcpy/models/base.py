"""
Base model classes used across kMCpy.
"""
from abc import ABC, abstractmethod
import json
import logging

from kmcpy.models.fitting.registry import get_fitter_for_model

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class BaseModel(ABC):
    """
    Base class for models in kmcpy.
    
    This abstract class provides a common interface for model objects,
    including serialization, deserialization, and computation methods. Subclasses must implement all
    abstract methods to define specific model behavior.
    
    Attributes:
        name (str, optional): Name of the model instance.
    """
    fitter_class = None

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseModel. This method can be overridden by subclasses to handle specific initialization.
        """
        self.name = kwargs.get("name", None)

    @classmethod
    def get_fitter_class(cls):
        """Return fitter implementation for this model class."""
        fitter_class = get_fitter_for_model(cls)
        if fitter_class is not None:
            return fitter_class
        if cls.fitter_class is not None:
            return cls.fitter_class
        raise NotImplementedError(
            f"{cls.__name__} does not define a fitter_class and has no fitter "
            "registered in kmcpy.models.fitting.registry."
        )

    def fit(self, *args, **kwargs):
        """Fit model parameters using the model-specific fitter implementation."""
        fitter = self.__class__.get_fitter_class()()
        return fitter.fit(*args, **kwargs)

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
    
    @abstractmethod
    def build(self, *args, **kwargs):
        """
        Build the model based on the provided parameters.
        This method must be implemented by subclasses.
        
        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def as_dict(self):
        """
        Convert the model object to a dictionary representation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abstractmethod
    def from_json(cls, fname):
        """
        Load a model object from a JSON file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def to_json(self, fname):
        from kmcpy.io import convert
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
    """
    CompositeModel is an abstract base class for combining multiple models into a single composite model.
    
    This class provides a framework for managing a collection of models and defines abstract methods 
    that must be implemented by subclasses to perform computations, probability calculations, and 
    serialization/deserialization.
    
    Args:
        models (list): A list of model instances to be combined in the composite model.
        *args: Variable length argument list passed to the BaseModel.
        **kwargs: Arbitrary keyword arguments passed to the BaseModel.
    
    Attributes:
        models (list): The list of models included in the composite model.
    
    Note:
        Subclasses must implement the abstract methods to provide specific functionality for computation and serialization.
    """
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
