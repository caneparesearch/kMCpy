"""
Base model classes used across kMCpy.
"""
from abc import ABC, abstractmethod
import logging

import numpy as np
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

from kmcpy.models.fitting.registry import get_fitter_for_model
from kmcpy.models.registry import model_class_for_payload, model_class_for_type

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)


MODEL_FILETYPE = "kmcpy.model_file"
SUPPORTED_MODEL_FILETYPES = frozenset({MODEL_FILETYPE})


def require_model_file_payload(payload):
    """Validate and return a serialized model envelope dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Model file must be a JSON object")

    if payload.get("filetype") not in SUPPORTED_MODEL_FILETYPES:
        raise ValueError(
            f"Unsupported model filetype. Expected '{MODEL_FILETYPE}'."
        )

    return payload


def require_model_type(payload, model_type: str):
    """Validate that a serialized model envelope declares the expected type."""
    data = require_model_file_payload(payload)
    observed = data.get("model_type")
    if observed != model_type:
        raise ValueError(f"Expected model_type '{model_type}', got '{observed}'")
    return data


class BaseModel(MSONable, ABC):
    """
    Base class for models in kmcpy.
    
    This base class provides common serialization and loading conventions for
    model objects. Scientific operations such as ``compute``, ``build``, and
    ``compute_probability`` are optional because different model classes have
    different roles in a KMC workflow.

    Constructor convention (pymatgen-style):
    - `as_dict` and `from_dict` handle structured data.
    - `to` and `from_file` handle file I/O.
    
    Model files come in two forms, both handled by :meth:`from_file`:

    - the ``as_dict`` payload written by :meth:`to`, with ``@module`` and
      ``@class``;
    - an envelope ``{"filetype": "kmcpy.model_file", "model_type": MODEL_TYPE,
      PAYLOAD_KEY: payload}``. Subclasses set ``MODEL_TYPE`` and, when the
      payload is nested, ``PAYLOAD_KEY``.

    Attributes:
        name (str, optional): Name of the model instance.
    """
    fitter_class = None
    MODEL_TYPE: str | None = None
    PAYLOAD_KEY: str | None = None

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

    def initialize_state(
        self,
        *,
        simulation_state,
        event_lib=None,
        structure=None,
        config=None,
        active_site_order=None,
    ) -> None:
        """Initialize optional stateful model caches from the KMC state.

        Stateless models can ignore this hook. Stateful adapters can use it to
        build their own occupancy representation once, instead of rebuilding it
        during every event-rate evaluation.
        """
        return None

    def apply_event(self, *, event, simulation_state) -> None:
        """Commit an accepted event to optional model-side state.

        Stateless models can ignore this hook. Stateful external adapters should
        update only the changed sites here so their internal state stays aligned
        with kMCpy's ``State``.
        """
        return None

    @classmethod
    def from_config(cls, config):
        """Load the configured model.

        Called on ``BaseModel``, this dispatches to the concrete model class
        declared by the model file or ``config.model_type`` (see :meth:`load`).
        Called on a concrete subclass, it loads that subclass directly from
        ``config.model_file``.
        """
        if cls is not BaseModel:
            return cls.from_file(config.model_file)
        return BaseModel.load(
            getattr(config, "model_file", ""),
            model_type=getattr(config, "model_type", None),
        )

    @staticmethod
    def load(model_file, model_type: str | None = None) -> "BaseModel":
        """Load a model file of any registered model type.

        The class is taken from the file: the ``model_type`` of a
        ``kmcpy.model_file`` envelope, or the ``@module``/``@class`` of a
        payload written by :meth:`to`. ``model_type`` (default
        ``"composite_lce"``) is only used for files that carry neither.
        """
        payload = loadfn(model_file, cls=None)
        if isinstance(payload, dict) and "filetype" in payload:
            require_model_file_payload(payload)
            file_model_type = payload.get("model_type")
            if not isinstance(file_model_type, str) or not file_model_type.strip():
                raise ValueError("Model file must include a non-empty 'model_type'")
            return model_class_for_type(file_model_type).from_file(model_file)
        if isinstance(payload, dict) and "@module" in payload and "@class" in payload:
            model_class = model_class_for_payload(payload)
            if not callable(getattr(model_class, "from_file", None)):
                raise ValueError(
                    f"Serialized model class '{payload['@module']}."
                    f"{payload['@class']}' does not provide from_file()."
                )
            return model_class.from_file(model_file)
        return model_class_for_type(model_type or "composite_lce").from_file(model_file)

    def __str__(self):
        """Return a compact string representation."""
        return self.__repr__()
    
    def __repr__(self):
        """Return a compact debug representation."""
        name = getattr(self, "name", None)
        if name is None:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(name={name!r})"
    
    def compute(self, *args, **kwargs):
        """Compute this model's native quantity, when the model defines one."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute()."
        )
    
    def compute_probability(self, *args, **kwargs):
        """Compute an event rate/probability for KMC, when supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_probability(). "
            "Use a KMC rate model such as CompositeLCEModel or LocalBarrierModel."
        )

    def compute_probabilities(
        self,
        *,
        events,
        event_indices,
        runtime_config,
        simulation_state,
    ) -> np.ndarray:
        """Compute rates in Hz for ``events[i]`` for each ``i`` in ``event_indices``.

        KMC calls this after every accepted event to refresh the dependent
        event rates. The default evaluates ``compute_probability`` one event at
        a time; models can override it with a batched implementation that
        returns the same values.
        """
        return np.array(
            [
                self.compute_probability(
                    event=events[event_index],
                    runtime_config=runtime_config,
                    simulation_state=simulation_state,
                )
                for event_index in event_indices
            ],
            dtype=np.float64,
        )
    
    def build(self, *args, **kwargs):
        """Build model data from scientific inputs, when supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement build()."
        )
    
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
    def from_file(cls, fname):
        """Create a model object from a serialized file or model-file envelope."""
        logger.info("Loading %s from: %s", cls.__name__, fname)
        return cls.from_dict(loadfn(fname, cls=None))

    def to(self, fname, indent: int = 2):
        """Save the model's ``as_dict`` payload to a JSON file."""
        logger.info("Saving %s to: %s", self.__class__.__name__, fname)
        dumpfn(self.as_dict(), fname, indent=indent)

    @classmethod
    def _unwrap_model_file(cls, data):
        """Return the model payload from a model-file envelope, or ``data`` unchanged.

        Envelopes with ``filetype`` are validated against ``MODEL_TYPE``. For
        compatibility, an envelope without ``filetype`` is unwrapped when its
        ``model_type`` matches.
        """
        if not isinstance(data, dict) or cls.MODEL_TYPE is None:
            return data
        if "filetype" in data:
            require_model_type(data, cls.MODEL_TYPE)
        elif data.get("model_type") != cls.MODEL_TYPE or (
            cls.PAYLOAD_KEY is not None and cls.PAYLOAD_KEY not in data
        ):
            return data
        if cls.PAYLOAD_KEY is None:
            return data
        payload = data.get(cls.PAYLOAD_KEY)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{cls.__name__} model file is missing object key '{cls.PAYLOAD_KEY}'"
            )
        return payload
