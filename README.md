# Implementation of Spatial Configuration U-net Using PyTorch

The architecture was inspired by [Integrating spatial configuration into heatmap regression based CNNs for landmark localization](https://www.sciencedirect.com/science/article/pii/S1361841518305784).

---

## Overview

### Model

The model is implemented using PyTorch, following the architecture shown below.

<p align="center">
    <img src="docs/Spatial configuration-U-net.jpg" width="750"/>
</p>

### Data

A portion of hand radiographs from the [Image Processing and Informatics Lab](https://ipilab.usc.edu/research/baaweb/) is used for the application example. The data has been downloaded and preprocessed.

The preprocessed data for training contains 91 images of size 256x256. Data augmentation includes intensity adjustments, translations, rotations, size scaling, and histogram matching.

See `data_preprocessing.py` and `data_loader.py` for details.

### Dependencies

* Torch 2.3.1+cu118
* PIL 10.2.0

This code is written with Python 3.10



