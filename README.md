# Prospective Failure Time Forecasting (PFTF)

PFTF is a temporal forecasting model for rock slope failures and other landslides proposed by [Leinauer et al. (2023)](https://www.nature.com/articles/s43247-023-00909-z). 


## Status Quo & Road Map

### Status Quo
This repository is under active development. 
As of now, it contains a proof of concept of the PFTF method, implemented in Python.

### Road Map
Currently, the following aspects are under active development:

* improve output plots (e.g. resolution and aspect ratio, standalone frontend)
* add ability to process live data
* reimplement core functionality in Cython
* improve documentation
* unit tests





## Usage

### Document Structure

#### Configuration Files
Use the `.ini` configuration files located in `/configs` to specify input and output file locations, smoothing and velocity window lengths and more.

#### Input Data
The `/input` directory contains (synthetic) relative displacement time series of different temporal resolution in `.csv` format, that may be used for demonstration purposes.
Adapt the configuration file (see above) to use your own files.

#### Output
By default, PFTF writes its outputs, various plots and CSV files (per smoothing window) representing the model's internal state, into `/output`.

### Setup
1. Clone or download repository and navigate to respective directory in a Unix or Linux shell
2. Create and activate a Python virtual environment and install project requirements
    ```
    python3 -m venv <name>
    source <name>/bin/activate
    pip install -r requirements.txt
    ```


3. Run PFTF in simulation mode with 
    ```
    cd source
    python main.py
    ```
    Note that `main.py` is the program's entry point!
4. Find the resulting plots and raw data in `/output`
