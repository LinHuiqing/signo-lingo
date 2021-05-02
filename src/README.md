# Model-Related Code

This directory contains all code related to the model, including developing, testing and data visualisation.

## 1. Table of contents <a name="TOC"></a>

1. [Table of contents](#TOC)
2. [Directory structure](#DS)
3. [Final files and instructions on running them](#INSTRUCTIONS)

## 2. Directory structure <a name="DS"></a>

```utf-8
.
├── data/                     # csvs for the 10 sampled classes
├── dataset/                  # full AUTSL dataset
├── dev_model/                # code for development of model
├── test_model/               # code for testing of model
├── environment.yml           # environment file for conda
├── log_file_plots.ipynb      # code for plotting graphs based on log files
├── requirements.txt          # relavant pip packages
├── sample_data_classes.py    # code for extracting a random subset of classes
└── README.md
```

## 3. Final files and instructions on running them <a name="INSTRUCTIONS"></a>

There are 2 ways of setting up:

1. Using `conda`
2. Using `pip`

### Using `conda`

1. Configure the conda enviornment
   ```bash
   conda env create -f environment.yml
   ```
2. To use environment
   ```bash
   conda activate dl-big-proj
   ```
3. Install `pip` on conda environment
   ```bash
   conda install pip
   ```
4. Install `torchinfo` with `pip`
   ```bash
   pip install torchinfo
   ```
5. Open jupyter noteboook
   ```bash
   jupyter-notebook
   ```
6. To exit environment
   ```bash
   conda deactivate
   ```
7. To destroy environment
   ```bash
   conda info --envs
   conda env remove --name dl-big-proj
   ```

### Using `pip`

1. Create a virtual environment
   ```bash
   virtualenv venv
   ```
2. To use environment
   ```bash
   source venv/bin/activate
   ```
3. Install relevant packages
   ```bash
   pip install -r requirements.txt
   ```
4. Open jupyter noteboook
   ```bash
   jupyter-notebook
   ```
5. To exit environment
   ```bash
   deactivate
   ```
6. To destroy environment
   ```bash
   rm -r venv
   ```
