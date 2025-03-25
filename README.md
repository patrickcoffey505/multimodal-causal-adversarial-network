# multimodal-causal-adversarial-network

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The goal of this project is to characterize the dynamic bidirectional effective connectivity among distributed brain regions during visual categorization by leveraging high-frequency (beta/gamma band) EEG signals and simultaneous fMRI recordings through an enhanced multimodal causal adversarial network architecture.

This repository contains an implementation of the MCAN model in TensorFlow 2.x. The model learns dynamic effective connectivity from multimodal neuroimaging data (fMRI and EEG) using an adversarial framework.

The data can be found here: https://www.dropbox.com/scl/fo/ixtf7zwh4066fcc3evdfn/AD8-HENbfDjILdOSafV4qN4?rlkey=5w0a3lgumhroshdn1kxbbo1ov&e=1&dl=0

## Requirements

TODO

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         multimodal_causal_adversarial_network and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── multimodal_causal_adversarial_network   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes multimodal_causal_adversarial_network a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

