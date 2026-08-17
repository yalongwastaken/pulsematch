# PulseMatch

PulseMatch is a deep learning-based approach for optimizing matched filters in digital communication systems. The project focuses on learning FIR filter taps for a matched filter that correlates with a given pulse shaping waveform. By primarily leveraging CNNs, PulseMatch aims to determine the optimal filter taps for any unknown, pulse-shaped signal based on received signal data, enhancing detection accuracy and overall system performance in varying channel conditions.

> **Status: complete.**

## Data

The dataset used for training and evaluation is entirely synthesized, primarily due to the lack of openly available signal data that aligns with the niche goals of this project. The synthetic data encompasses diverse pulse shaping waveforms (defined by FIR filter taps), parameterized by different, applicable characteristics. This approach ensures flexibility in training and testing while aiming to create/evaluate a model that generalizes well across various digital communication scenarios.

## Repository Structure
``` 
├── notebooks/                     # Jupyter/Colab notebooks for experiments, prototyping, and analysis.
├── src/                           # Source code for core functionality, model definitions, and utilities.
│   └── data/                      # Contains all generated (synthesized) data.
│   └── data_generation/           # Tools for synthetic data generation and signal simulation.
│   └── utils/                     # Utility functions (e.g. data loading, preprocessing).
│   └── cnn.py                     # Definition and training logic for the 1D CNN.
├── supplementary_results/         # Additional plots and extended evaluation outputs.
├── ...                            # Configuration files, deployment scripts, requirements, and other setup files.
└── README.md                      # Project overview, setup instructions, and usage guidelines.
``` 

## Author

**Anthony Yalong**
- Email: yalong.anthony123@gmail.com
- GitHub: [@yalongwastaken](https://github.com/yalongwastaken)
