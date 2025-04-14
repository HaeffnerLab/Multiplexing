#!/bin/bash

# Create a new conda environment named "multiplexing" with Python 3.9
conda create -y -n multiplexing python=3.9

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate multiplexing

# Install packages from requirements.txt
conda install -y -c conda-forge numpy=1.20.3 scipy=1.7.3 matplotlib=3.5.3 h5py=3.6.0
conda install -y -c conda-forge qutip=4.7.0 pynverse=0.1.4 pandas=1.3.5

# Install Jupyter
conda install -y -c conda-forge jupyter notebook

# Create a kernel specification for Jupyter
python -m ipykernel install --user --name multiplexing --display-name "Python (multiplexing)"

echo "Conda environment 'multiplexing' has been created and configured."
echo "To activate the environment, run: conda activate multiplexing"
echo "To use the environment in Jupyter, select the 'Python (multiplexing)' kernel." 