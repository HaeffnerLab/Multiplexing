# Multiplexing - Ion Shuttling Analysis

This repository contains analysis of 9-ion shuttling Rabi oscillation data (dated January 16, 2025).

## Setup Instructions

### Prerequisites
- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Environment Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Multiplexing
   ```

2. Set up the conda environment using the provided script:
   ```bash
   chmod +x setup_multiplexing_env.sh
   ./setup_multiplexing_env.sh
   ```

   This script will:
   - Create a conda environment named "multiplexing" with Python 3.9
   - Install all required dependencies
   - Set up a Jupyter kernel for the environment

3. Alternatively, you can manually create the environment:
   ```bash
   conda create -y -n multiplexing python=3.9
   conda activate multiplexing
   conda install -y -c conda-forge --file requirements.txt
   conda install -y -c conda-forge jupyter notebook
   python -m ipykernel install --user --name multiplexing --display-name "Python (multiplexing)"
   ```

## Usage

1. Activate the conda environment:
   ```bash
   conda activate multiplexing
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the "Shuttling 9-ion rabi 20250116.ipynb" notebook

4. Select the "Python (multiplexing)" kernel from the kernel menu

## Dependencies

The analysis requires the following Python packages (specified in requirements.txt):
- numpy 1.20.3
- scipy 1.7.3
- matplotlib 3.5.3
- qutip 4.7.0
- h5py 3.6.0
- pynverse 0.1.4
- pandas 1.3.5

## License

[Specify your license here] 