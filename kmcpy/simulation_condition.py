from abc import ABC, abstractmethod
from typing import Union
from kmcpy.model.model import BaseModel, CompositeKMCModel

class SimulationCondition(ABC):
    """
    Class for KMC simulation conditions.
    """
    def __init__(self, name: str, temperature: float, attempt_frequency: float):
        """
        Initialize the KMC condition with a name, temperature, and attempt frequency.

        :param name: Name of the condition.
        :param temperature: Temperature in Kelvin.
        :param attempt_frequency: Attempt frequency in Hz.
        """
        name = 'SimulationCondition' if name is None else name
        self.name = name
        self.temperature = temperature
        self.attempt_frequency = attempt_frequency

    def get_condition(self) -> str:
        return f"{self.name}: T={self.temperature}K, f={self.attempt_frequency}Hz"
    
class SimulationConfig(SimulationCondition):
    from kmcpy.event import EventLib

    """
    Class for simulation configuration.
    """
    def __init__(
        self,
        name: str,
        model: Union[BaseModel, CompositeKMCModel],
        event_lib: EventLib,
        simulation_state: "SimulationState",
        temperature: float = 300.0,
        attempt_frequency: float = 1e13
    ):
        super().__init__(name, temperature, attempt_frequency)
        self.model = model
        self.event_lib = event_lib
        self.simulation_state = simulation_state

    def get_condition(self) -> str:
        return f"{super().get_condition()}, Event Library:{self.event_lib}, Simulation State:{self.simulation_state}"
    
class SimulationState(ABC):
    def __init__(self, occupations:list=[], probabilities:list=[], barriers:list=[], time=0.0, step=0):
        """
        occupations: dict or array-like mapping site index -> species/occupation
        time: current simulation time
        step: current step count
        """
        self.occupations = occupations
        self.time = time
        self.step = step
        self.probabilities = probabilities
        self.barriers = barriers

    @abstractmethod
    def update(self, update_message:dict, indices:list):
        raise NotImplementedError(
            "This method should be implemented in the subclass to update the state based on the update message."
        )
    
    def advance_time(self, dt):
        self.time += dt
        self.step += 1

    def to(self, filename: str) -> None:
        """
        Save the state to a file.
        
        :param filename: The name of the file to save the state.
        """
        if filename.endswith('.json'):
            import json
            with open(filename, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        elif filename.endswith('.h5'):
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, value in self.__dict__.items():
                    f.create_dataset(key, data=value)
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
    @classmethod
    def from_file(cls, filename: str) -> "SimulationState":
        """
        Load the state from a file.
        
        :param filename: The name of the file to load the state from.
        :return: An instance of SimulationState.
        """
        if filename.endswith('.json'):
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
        elif filename.endswith('.h5'):
            import h5py
            with h5py.File(filename, 'r') as f:
                data = {key: f[key][()] for key in f.keys()}
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
        return cls(**data)

class LCESimulationState(SimulationState):
    def __init__(self, occupations:list=[], probabilities:list=[], barriers:list=[],  ekra:list=[], 
                 esite_diff:list=[], time=0.0, step=0):
        super().__init__(occupations, probabilities, barriers, time, step)
        self.ekra = ekra
        self.esite_diff = esite_diff

    def update(self, update_message:dict, indices:list):
        """
        Update the occupations and barriers based on the update message.
        
        :param update_message: Dictionary containing 'ekra', 'esite_diff', 'direction', 'barrier', 'probability'.
        :param indices: List of indices to update in the occupations.
        """
        ekra = update_message.get('ekra')
        esite_diff = update_message.get('esite_diff')
        barrier = update_message.get('barrier')
        probability = update_message.get('probability')

        for idx in indices:
            self.occupations[idx] *= -1
            self.barriers[idx] = barrier
            self.probabilities[idx] = probability
            self.ekra[idx] = ekra
            self.esite_diff[idx] = esite_diff