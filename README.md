# PulseMatch

## Description

PulseMatch is a deep learning-based approach for optimizing matched filters in digital communication systems. The project focuses on learning FIR filter taps for a matched filter that correlates with a given pulse shaping waveform, with the objective of maximizing the signal-to-noise ratio (SNR). By leveraging CNNs, LSTMs, and Transformers, PulseMatch aims to determine the optimal filter taps for any unknown, pulse-shaped signal based on received signal data, enhancing detection accuracy and overall system performance in varying channel conditions.

## Data

The dataset used for training and evaluation is entirely synthesized, primarily due to the lack of openly available signal data that aligns with the niche goals of this project. The synthetic data encompasses diverse pulse shaping waveforms (defined by FIR filter taps), various modulation schemes, different sample-per-symbol (SPS) rates, and realistic channel noise models. This approach ensures flexibility in training and testing while aiming to create a model that generalizes well across various digital communication scenarios.

## About

The `PulseMatch` project is developed by Anthony Yalong, a Computer Engineering student at The George Washington University, with a background in machine learning, digital signal processing, and embedded systems. The project is driven by a passion for applying deep learning techniques and ideas to areas related to computer engineering topics. PulseMatch aims to push the boundaries of adaptive filter design using AI-driven approaches.