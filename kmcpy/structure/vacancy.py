from pymatgen.core import DummySpecies

class Vacancy(DummySpecies):
    """
    A dummy species to represent a vacancy in the lattice model.
    """
    def __init__(self):
        super().__init__('X', 0)  # 'X' is a common symbol for vacancies
        self.is_vacancy = True

    def __repr__(self):
        return "Vacancy()"