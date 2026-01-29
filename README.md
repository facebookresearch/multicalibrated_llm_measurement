# Multicalibration for Prevalence Measurement

This repository contains exploratory analysis demonstrating the role of multicalibration in model-based prevalence estimation using the American Community Survey (ACS) dataset.

## Setup

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n mc_measurement python=3.10 -y
conda activate mc_measurement

# Install requirements
pip install -r requirements.txt

# Install MCGrad from GitHub (latest version)
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"

# Register the kernel with Jupyter
python -m ipykernel install --user --name=mc_measurement --display-name="Python (mc_measurement)"
```

### Option 2: Using venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install MCGrad from GitHub
pip install "MCGrad[tutorials] @ git+https://github.com/facebookincubator/MCGrad.git"

# Register the kernel with Jupyter
python -m ipykernel install --user --name=mc_measurement --display-name="Python (mc_measurement)"
```

## Running the Analysis

1. Activate your environment:
   ```bash
   conda activate mc_measurement  # or: source venv/bin/activate
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `acs_exploration.ipynb` and select the "Python (mc_measurement)" kernel

4. Run all cells. The first run will download ACS data (~500MB) to the `data/` directory.

## Repository Structure

```
mc_measurement/
├── acs_exploration.ipynb    # Main analysis notebook
├── mcgrad_tutorial_notebook.ipynb  # Reference MCGrad tutorial
├── helpers.py               # Helper functions for data loading and analysis
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── data/                   # Downloaded ACS data (not tracked in git)
```

## Data

The analysis uses the American Community Survey (ACS) data via the `folktables` package. Data is automatically downloaded on first run and cached in the `data/` directory.

- Training states: TX, MI, PA, OH, IL, GA, NC, VA
- OOD (held-out) states: CA, NY, FL, WA, AZ, CO
- Survey years: 2016, 2017, 2018

## Key Findings

See the notebook for detailed analysis. The main insights are:

1. MCGrad effectively reduces multicalibration error (MCE) compared to isotonic regression
2. For prevalence estimation, the choice of segment features matters:
   - MCGrad only improves bias for segments it's explicitly calibrated on
   - Adding STATE as a segment feature reduces state-level bias (in-distribution)
   - SEX-level calibration transfers to out-of-distribution states
