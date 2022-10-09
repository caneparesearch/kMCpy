# Installation


## User

1. Create a conda environment
`conda create -n kmcpy`
`conda activate kmcpy`
2. Install pip
`conda install python=3.8`
3. Install required packages and kmcPy
`pip install -r requirement.txt .`

## Developer

1. Create a conda environment
`conda create -n kmcpy`
`conda activate kmcpy`
2. Install pip
`conda install python=3.8`
3. For developer, use editable mode (developer mode) of pip
`pip install -r requirement.txt -e .`
4. For maintainer of document, install the dependency for generating the documentation
`brew install pandoc`
`pip install -r doc/doc_requirements.txt`
