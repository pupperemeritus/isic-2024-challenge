# ISIC 2024 Challenge: LesionNet for Skin Lesion Classification

This repository contains the modified PyTorch Lightning implementation of GuruNet from PET model, a deep learning model for classifying skin lesions using image and tabular metadata for the ISIC 2024 Challenge.

## Key Features

- **LesionNet Model:** A custom PyTorch Lightning model designed for skin lesion classification, integrating image and metadata features.

- **Partial AUROC:** Taken from the official ISIC 2024 Challenge evaluation metric.

- **Data Handling:** The project's data handling involves separate processing of image data (from HDF5 files) and tabular metadata (from CSV files). The metadata undergoes cleaning, standardization, and categorical encoding before being combined with image features. The GuruNet model, originally designed for PET data classification, has been adapted to handle this hybrid input by incorporating distinct pathways: a convolutional neural network processes the image data, while fully connected layers handle the tabular metadata. These two processed streams are then concatenated in "hybrid input layers" before the final classification, allowing the model to leverage both modalities.

- **Model Export:** Supports exporting trained models to the `.safetensors` format.

## Setup

1. Clone this repository.

2. Install dependencies (see `requirements.txt` below).

3. Place your `train-image.hdf5`, `test-image.hdf5`, `train-metadata.csv`, and `test-metadata.csv` files in the project root.

## Usage

Run `competition-draft-pytorch.ipynb` to train and evaluate the model. Use `compile_model.py` to convert checkpoints to `.safetensors`.
