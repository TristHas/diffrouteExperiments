# DiffRoute Experiment Reproduction

## Introduction

This repository contains a snapshot of the code used in the DiffRoute model paper at the time of submission.  
**Important:** this is an **old version** of the code. Please use it **only** to reproduce the exact results presented in the paper.  
It is provided strictly for **reproducibility purposes**.

For all other purposes, please follow the instructions in the main **DiffRoute** and **DiffHydro** repositories.  
In particular, the experiments of the paper have been integrated into easy-to-use example notebooks illustrating typical usage within the DiffHydro library.  
Those notebooks use the most up-to-date version of the model, with significant improvements in GPU memory footprint, computation speed, and a polished API for easier use.

---

## Installation and Usage

### DiffRoute Installation

This code uses the `paper` branch of the DiffRoute repository.

To install:

    git clone git@github.com:TristHas/DiffRoute.git
    cd diffroute
    git fetch
    git checkout paper
    pip install -e .

### Additional Dependencies

You will need `huggingface_hub` to download the GEOGloWS dataset:

    pip install huggingface_hub

### Download experiment repo

Once diffroute and optional dependencies are installed, download the current repository

    git clone git@github.com:TristHas/diffrouteExperiments.git
    cd diffrouteExperiments

and open the relevant jupyter notebooks

---

## Data Download

To run the GEOGloWS notebooks, download the dataset by executing:

    python geoglows/download_dataset.py

Notes:

- The dataset requires **more than 700 GB** of disk space.
- By default, the target directory for the dataset download is `data/geoglows`.
- If you do not have enough free space on the current partition, set the variable `DATA_ROOT` in `geoglows/config.py` to a path with at least **700 GB** of free disk space.
