"""
Base model classes used across kMCpy.
"""
from abc import ABC, abstractmethod
import importlib
import logging

from monty.serialization import loadfn

from kmcpy.io.registry import MODEL_CLASS_REGISTRY
from kmcpy.models.fitting.registry import get_fitter_for_model
from kmcpy.models.schema import MODEL_FILETYPE, require_model_file_payload

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class BaseModel(ABC):
    """
    Base class for models in kmcpy.
    
    This abstract class provides a common interface for model objects,
    including serialization, deserialization, and computation methods.

    Constructor convention (pymatgen-style):
    - `as_dict` and `from_dict` handle structured data.
    - `to` and `from_file` handle file I/O.
    
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

    @classmethod
    def from_config(cls, config):
        """Load the configured model.

        Called on ``BaseModel``, this dispatches to the concrete model class
        declared by the model file or ``config.model_type``. Called on a
        concrete subclass, it loads that subclass directly from
        ``config.model_file``.
        """
        if cls is not BaseModel:
            return cls.from_file(config.model_file)

        model_file = getattr(config, "model_file", "")
        model_type = None
        if model_file:
            payload = loadfn(model_file, cls=None)
            if isinstance(payload, dict) and "filetype" in payload:
                require_model_file_payload(payload)
                if payload.get("filetype") == MODEL_FILETYPE:
                    model_type = payload.get("model_type")
                    if not isinstance(model_type, str) or not model_type.strip():
                        raise ValueError(
                            "Model file must include a non-empty 'model_type'"
                        )

        if model_type is None:
            model_type = getattr(config, "model_type", None) or "composite_lce"

        if model_type not in MODEL_CLASS_REGISTRY:
            available_types = list(MODEL_CLASS_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available types: {available_types}"
            )

        model_class_path = MODEL_CLASS_REGISTRY[model_type]
        module_path, class_name = model_class_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import model class '{model_class_path}': {e}")

        return model_class.from_config(config)

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
    def from_dict(cls, d):
        """
        Create a model object from a dictionary representation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abstractmethod
    def from_file(cls, fname):
        """
        Create a model object from a serialized file.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def to(self, fname):
        """
        Save the model object to a JSON file.
        """
        from monty.serialization import dumpfn

        logger.info("Saving model to: %s", fname)
        dumpfn(self.as_dict(), fname, indent=4)


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

    @classmethod
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
