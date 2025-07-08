from abc import ABC, abstractmethod

class ModelParameters(ABC):
    """
    Abstract base class for model parameters.
    """
    def __init__(self, name: str):
        """
        Initialize the model parameters with a name.
        
        :param name: Name of the model parameters.
        """
        self.name = name

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Get the model parameters as a dictionary.
        
        :return: Dictionary of model parameters.
        """
        pass

    def __str__(self) -> str:
        params = self.get_parameters()
        param_values_str = ', '.join(f'{key}={value}' for key, value in params.items())
        return f"{self.name} Parameters: {param_values_str}"
    
    def __getattr__(self, name):
        try:
            return self.get_parameters()[name]
        except KeyError:
            raise AttributeError(f"{self.name} object has no attribute '{name}'")
    
    def to(self, filename: str) -> None:
        """
        Save the parameters to a file.
        
        :param filename: The name of the file to save the parameters.
        """
        if filename.endswith('.json'):
            import json
            with open(filename, 'w') as f:
                json.dump(self.get_parameters(), f, indent=4)
        elif filename.endswith('.h5'):
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, value in self.get_parameters().items():
                    f.create_dataset(key, data=value)
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
class LCEModelParameters(ModelParameters):
    """
    Class for storing fitted results for LocalClusterExpansion Model.
    """
    def __init__(self, keci:list[float], empty_cluster:float, 
                 sublattice_indices:list[int], 
                 weight:list[float], alpha:float, time_stamp:float, time:str, 
                 rmse:float, loocv:float,**kwargs) -> None:
        super().__init__(name="LCEModelParameters")
        self.keci = keci
        self.empty_cluster = empty_cluster
        self.sublattice_indices = sublattice_indices
        self.weight = weight
        self.alpha = alpha
        self.time_stamp = time_stamp
        self.time = time
        self.rmse = rmse
        self.loocv = loocv

    def get_parameters(self) -> dict:
        return {
            "keci": self.keci,
            "empty_cluster": self.empty_cluster,
            "sublattice_indices": self.sublattice_indices,
            "weight": self.weight,
            "alpha": self.alpha,
            "time_stamp": self.time_stamp,
            "time": self.time,
            "rmse": self.rmse,
            "loocv": self.loocv
        }

    @classmethod
    def from_json(cls, filename: str) -> "LCEModelParameters":
        """
        Load parameters from a JSON file.
        
        :param filename: The name of the file to load the parameters from.
        """
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
            return cls(**data)
        
class LCEModelParamHistory(ABC):
    """
    Class for storing a history of LCE model parameters.
    """
    def __init__(self):
        self.history = []

    @classmethod
    def from_file(cls, filename: str) -> "LCEModelParamHistory":
        """
        Load the history from a file.
        
        :param filename: The name of the file to load the history from.
        """
        param_history = cls()
        if filename.endswith('.json'):
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
                param_history.history = [LCEModelParameters(**params) for params in data]
        elif filename.endswith('.h5'):
            import h5py
            with h5py.File(filename, 'r') as f:
                param_history.history = []
                for key in f.keys():
                    params = {k: v[()] for k, v in f[key].items()}
                    param_history.history.append(LCEModelParameters(**params))
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        return param_history

    def append(self, parameters: LCEModelParameters)-> None:
        """
        Add a set of parameters to the history.
        
        :param parameters: An instance of LCEModelParameters.
        """
        self.history.append(parameters)

    def get_history(self) -> list[LCEModelParameters]:
        """
        Get the history of parameters.
        
        :return: List of LCEModelParameters instances.
        """
        return self.history
    
    def to(self, filename: str)-> None:
        """
        Save the history to a file.
        
        :param filename: The name of the file to save the history.
        """
        if filename.endswith('.json'):
            import json
            with open(filename, 'w') as f:
                json.dump([params.get_parameters() for params in self.history], f, indent=4)
        elif filename.endswith('.h5'):
            import h5py
            with h5py.File(filename, 'w') as f:
                for i, params in enumerate(self.history):
                    grp = f.create_group(f'parameter_set_{i}')
                    for key, value in params.get_parameters().items():
                        grp.create_dataset(key, data=value)
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")