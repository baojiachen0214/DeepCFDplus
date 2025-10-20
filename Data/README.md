# Data Directory

This directory contains the dataset files required for training the DeepCFD+ models.

## Required Files

- `dataX.pkl`: Input data (geometry representations)
- `dataY.pkl`: Output data (velocity and pressure fields)

## Data Format

The data is stored in pickle format with the following structure:

- `dataX.pkl`: Contains the input geometries represented as signed distance functions (SDF)
- `dataY.pkl`: Contains the corresponding velocity (Ux, Uy) and pressure (p) fields

## Source

The original dataset can be downloaded from [Zenodo](https://zenodo.org/record/3666056).

## Usage

The training script (`main.py`) automatically loads these files from this directory by default.