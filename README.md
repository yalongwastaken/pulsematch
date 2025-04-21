# PulseMatch

## Description

PulseMatch is a deep learning-based approach for optimizing matched filters in digital communication systems. The project focuses on learning FIR filter taps for a matched filter that correlates with a given pulse shaping waveform. By primarily leveraging CNNs, PulseMatch aims to determine the optimal filter taps for any unknown, pulse-shaped signal based on received signal data, enhancing detection accuracy and overall system performance in varying channel conditions.

## Data

The dataset used for training and evaluation is entirely synthesized, primarily due to the lack of openly available signal data that aligns with the niche goals of this project. The synthetic data encompasses diverse pulse shaping waveforms (defined by FIR filter taps), parameterized by different, applicable characteristics. This approach ensures flexibility in training and testing while aiming to create/evaluate a model that generalizes well across various digital communication scenarios.

## Project Structure
``` 
├── notebooks/                     # Jupyter/Colab notebooks for experiments, prototyping, and analysis.
├── src/                           # Source code for core functionality, model definitions, and utilities.
│   └── data/                      # Contains all generated (synthesized) data.
│   └── data_generation/           # Tools for synthetic data generation and signal simulation.
│   └── utils/                     # Utility functions (e.g. data loading, preprocessing).
│   └── cnn.py                     # Definition and training logic for the 1D CNN.
├── supplementary_results/         # Additional plots, evaluation outputs, or extended results used in the main paper.
├── ...                            # Configuration files, deployment scripts, requirements, and other setup files.
└── README.md                      # Project overview, setup instructions, and usage guidelines.
``` 

## About

The `PulseMatch` project is developed by Anthony Yalong, a Computer Engineering student at The George Washington University, with a background in machine learning, digital signal processing, and embedded systems. The project is driven by a passion for applying deep learning techniques and ideas to areas related to computer engineering topics. PulseMatch aims to push the boundaries of adaptive filter design using AI-driven approaches.