from distutils.core import setup
import os

__version__ = '0.1dev'
src_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name='kmcPy',
    version=__version__,
    description='Kinetic Monte Carlo Simulation using Python (kmcPy)',
    author="Zeyu Deng",
    author_email="dengzeyu@gmail.com",
    maintainer='Zeyu Deng',
    maintainer_email="dengzeyu@gmail.com",
    install_requires=['numpy','scipy','pymatgen','numba','tables','pandas'],
    license='MIT License',
    long_description=open('README.md').read(),
    python_requires='>=3.8',
)