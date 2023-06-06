from distutils.core import setup
import os

__version__ = '0.1dev'
src_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name='kMCpy',
    version=__version__,
    description='Kinetic Monte Carlo Simulation using Python (kMCpy)',
    author="Zeyu Deng",
    author_email="dengzeyu@gmail.com",
    maintainer='Zeyu Deng',
    maintainer_email="dengzeyu@gmail.com",
    install_requires=['pymatgen','numba','joblib','scikit-learn','glob2'],
    license='MIT License',
    long_description=open('README.md').read(),
    python_requires='>=3.8',
    package_dir={"kmcpy":'kmcpy'}
)
