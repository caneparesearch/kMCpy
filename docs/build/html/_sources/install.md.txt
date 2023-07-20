# Installation

## With GUI enabled (Recommended for Windows, Macos, Linux personal computer)

```
conda create -n kmcpy python wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt .
```

## With no GUI enabled (with access to command line environment)

```
conda create -n kmcpy python
conda activate kmcpy
pip install -r requirement.txt .
```

## For developers and building documentation

```
conda create -n kmcpy wxpython -c conda-forge
conda activate kmcpy
pip install -r requirement_gui.txt -e .
cd docs
pip install -r doc_requirements.txt
cd ..
python build_doc.py
```