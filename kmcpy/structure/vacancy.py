from pymatgen.core import DummySpecies

class Vacancy(DummySpecies):
    """
    Represents a vacancy site in a lattice model as a dummy species.

    This class inherits from DummySpecies and is used to indicate the presence of a vacancy
    (i.e., an unoccupied site) in the lattice. The vacancy is typically denoted by the symbol 'X'
    and has a charge of 0.

    Attributes
    ----------
    is_vacancy : bool
        Flag indicating that this species represents a vacancy.

    Examples
    --------
    >>> v = Vacancy()
    >>> print(v)
    Vacancy()
    """    
    def __init__(self):
        super().__init__('X', 0)  # 'X' is a common symbol for vacancies
        self.is_vacancy = True

    def __repr__(self):
        return "Vacancy()"
