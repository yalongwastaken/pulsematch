# Author: Anthony Yalong
# Description: This file generates the data for PulseMatch.

# imports
import numpy as np
import scipy.signal as signal
from typing import Tuple

# Modulation
modulation_types = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
]

# Samples per symbol (SPS)
sps_rates = [
    2,
    4,
    8,
    16,
]

# TODO: Look into noise levels. Current values are arbitrary.
noise_levels = [
    0.01,
    0.1,
    0.5,
    1.0,
]

def generate_bitstream(num_bits: int=None) -> np.ndarray:
    """
    Generate a random bitstream of given length.
    """
    if num_bits is None:
        num_bits = np.random.uniform(100, 10000)
        num_bits = int(num_bits)
    
    return np.random.randint(0, 2, num_bits)

# TODO: Implement modulation schemes.
def apply_modulation(bitstream: np.ndarray, modulation_type: str=None) -> Tuple[np.ndarray, str]:
    """
    Apply modulation to the bitstream.
    """
    if modulation_type is None:
        modulation_type = np.random.choice(modulation_types)

    if modulation_type == "BPSK":
        modulated_signal = 2 * bitstream - 1
    
    # TODO: Implement QPSK
    elif modulation_type == "QPSK":
        modulated_signal = 2 * bitstream - 1

    # TODO: Implement 8PSK
    elif modulation_type == "8PSK":
        modulated_signal = 2 * bitstream - 1

    # TODO: Implement 16QAM
    elif modulation_type == "16QAM":
        modulated_signal = 2 * bitstream - 1

    return modulated_signal, modulation_type

def upsample_signal(signal: np.ndarray, sps: int=None) -> Tuple[np.ndarray, int]:
    """
    Upsample the signal based on the selected SPS rate.
    """
    if sps is None:
        sps = np.random.choice(sps_rates)

    upsampled_signal = np.repeat(signal, sps)
    return upsampled_signal, sps

# TODO: Add RRC and RC filter taps generation.
def generate_fir_filter_taps() -> np.ndarray:
    """
    Generate FIR filter taps arbitrarily for pulse shaping.
    """
    num_taps = int(np.random.normal(64, 32))
    num_taps = np.clip(num_taps, 4, 513)

    # Generate random FIR filter taps
    filter_taps = np.random.uniform(-10, 10, num_taps)
    return filter_taps

def apply_fir_filter(signal: np.ndarray, filter_taps: np.ndarray=None) -> np.ndarray:
    """
    Apply FIR filter to the signal.
    """
    if filter_taps is None:
        filter_taps = generate_fir_filter_taps()

    filtered_signal = signal.lfilter(filter_taps, 1.0, signal)
    return filtered_signal, filter_taps
   
def apply_noise(signal: np.ndarray, noise_level: float=None) -> Tuple[np.ndarray, float]:
    """
    Apply noise to the signal.
    """
    if noise_level is None:
        noise_level = np.random.choice(noise_levels)

    noise = np.random.normal(0, noise_level, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, noise_level

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize the signal.
    """
    return signal / np.max(np.abs(signal))

# TODO: Implement data generation function.
# TODO: Implement HDF5 file generation and storage.
def generate_data(dataset_size: int=100000) -> None:
    """
    Generate the data for PulseMatch.
    """
    # Generate a random bitstream
    bitstream = generate_bitstream()

    # Apply modulation
    modulated_signal, modulation_type = apply_modulation(bitstream)

    # Upsample the signal
    upsampled_signal, sps_rate = upsample_signal(modulated_signal)

    # Apply the pulse shaping filter
    filtered_signal, filter_taps = apply_fir_filter(upsampled_signal)

    # Apply noise
    noisy_signal, noise_level = apply_noise(filtered_signal)

    # Normalize the signal
    normalized_signal = normalize_signal(noisy_signal)