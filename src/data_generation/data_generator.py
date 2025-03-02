# Author: Anthony Yalong
# Description: This file generates synthetic data for PulseMatch, a model designed to learn and predict pulse shaping functions
#              used in digital communication systems. The process involves the generation of a random bitstream, modulation,
#              upsampling, pulse shaping via FIR filters, and the addition of noise. The generated data is in IQ format (In-phase 
#              and Quadrature), which is commonly used in digital communication systems for representing complex signals.
#
# TODO: Implement HDF5 file generation and storage.

# Imports
import modulation
import common_filters
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple

DATASET_SIZE = 1000000
BITSTREAM_SIZE_MIN = 1024
BITSTREAM_SIZE_MAX = 65536

modulation_types = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "32QAM",
    "64QAM",
]

sps_rates = [
    2,
    4,
    8,
    16,
    32,
]

known_filters = [
    "RRC",
    "RC",
    "Gaussian",
    "Sinc",
]

windowing_functions = [
    "hamming",
    "hanning",
    "blackman",
    "bartlett",
]

noise_levels = [
    0,
    0.001,
    0.01,
    0.1,
    0.2,
    0.5,
    1.0,
]

def generate_bitstream(num_bits: int=None) -> Tuple[np.ndarray, int]:
    """
    Generate a random bitstream of given length.
    
    Parameters:
    - num_bits (int): Length of the bitstream. If None, a random length between BITSTREAM_SIZE_MIN and BITSTREAM_SIZE_MAX is chosen.

    Returns:
    - bitstream (np.ndarray): Randomly generated bitstream of 0s and 1s.
    - num_bits (int): Length of the generated bitstream.
    """
    if num_bits is None:
        num_bits = np.random.uniform(BITSTREAM_SIZE_MIN, BITSTREAM_SIZE_MAX)
        num_bits = int(num_bits)
    
    return np.random.randint(0, 2, num_bits), num_bits

def apply_modulation(bitstream: np.ndarray, modulation_type: str=None) -> Tuple[np.ndarray, str]:
    """
    Apply modulation to the bitstream.

    Parameters:
    - bitstream (np.ndarray): Input bitstream.
    - modulation_type (str): Type of modulation to apply. If None, a random modulation type is chosen.

    Returns:
    - modulated_signal (np.ndarray): Modulated signal.
    - modulation_type (str): Type of modulation applied.
    """
    if modulation_type is None:
        modulation_type = np.random.choice(modulation_types)

    # BPSK modulation
    if modulation_type == "BPSK":
        modulated_signal = modulation.modulate_bpsk(bitstream)
    
    # QPSK modulation
    elif modulation_type == "QPSK":
        modulated_signal = modulation.modulate_qpsk(bitstream)

    # 8PSK modulation
    elif modulation_type == "8PSK":
        modulated_signal = modulation.modulate_8psk(bitstream)
        
    # 16QAM modulation
    elif modulation_type == "16QAM":
        modulated_signal = modulation.modulate_16qam(bitstream)

    # 32QAM modulation
    elif modulation_type == "32QAM":
        modulated_signal = modulation.modulate_32qam(bitstream)

    # 64QAM modulation
    elif modulation_type == "64QAM":
        modulated_signal = modulation.modulate_64qam(bitstream)

    return modulated_signal, modulation_type

def generate_known_filter(modulation_type: str) -> Tuple[np.ndarray, str]:
    """
    Generate a known FIR filter (RRC, RC, etc.).
    
    Parameters:
    - modulation_type (str): Type of modulation used.

    Returns:
    - filter_taps (np.ndarray): Coefficients of the FIR filter.
    - filter_name (str): Name of the filter.
    """
    known_filter = np.random.choice(known_filters)

    # Root Raised Cosine (RRC) filter
    if known_filter == "RRC":
        return common_filters.rrc(modulation_type), "RRC"
    
    # Raised Cosine (RC) filter
    elif known_filter == "RC":
        return common_filters.rc(modulation_type), "RC"
    
    # Gaussian filter
    elif known_filter == "Gaussian":
        return common_filters.gaussian(modulation_type), "GAUSSIAN"
    
    # Sinc filter
    elif known_filter == "Sinc":
        return common_filters.sinc(modulation_type), "SINC"

def apply_upsampling(signal: np.ndarray, sps: int=None) -> Tuple[np.ndarray, int]:
    """
    Apply upsampling to the modulated signal.
    
    Parameters:
    - signal (np.ndarray): Input modulated signal.
    - sps (int): Symbol rate. If None, a random symbol rate is chosen.

    Returns:
    - signal (np.ndarray): Upsampled signal.
    - sps (int): Symbol rate used for upsampling.
    """
    if sps is None:
        sps = np.random.choice(sps_rates)

    # Upsample the signal by a random factor
    signal = np.repeat(signal, sps, axis=0)

    return signal, sps

def generate_random_filter(sps: int=None) -> Tuple[np.ndarray, str]:
    """
    Generate a random FIR filter.
    
    Parameters:
    - sps (int): Symbol rate.

    Returns:
    - filter_taps (np.ndarray): Coefficients of the FIR filter.
    - filter_name (str): Name of the filter.
    """
    # Define multipliers
    multipliers = np.arange(4, 10, 1)

    # Determine the number of taps based on the symbol rate (SPS)
    num_taps = sps * np.random.choice(multipliers)

    # Restrict the number of taps to a range (16, 512)
    num_taps = np.clip(num_taps, 16, 513)

    filter_taps = np.random.uniform(-1, 1, num_taps)

    # Apply a random windowing function
    window_type = np.random.choice(windowing_functions)

    # Hamming window
    if window_type == "hamming":
        window = np.hamming(num_taps)

    # Hanning window
    elif window_type == "hanning":
        window = np.hanning(num_taps)

    # Blackman window
    elif window_type == "blackman":
        window = np.blackman(num_taps)

    # Bartlett window
    elif window_type == "bartlett":
        window = np.bartlett(num_taps)

    filter_taps *= window

    # Normalize the filter taps to have unit energy, mimicking practical pulse shaping filters
    filter_taps /= np.linalg.norm(filter_taps, 2)

    return filter_taps, "RANDOM"

def apply_filter(signal: np.ndarray, filter_taps: np.ndarray) -> np.ndarray:
    """
    Apply the FIR filter to the modulated signal.
    
    Parameters:
    - signal (np.ndarray): Input modulated signal.
    - filter_taps (np.ndarray): Coefficients of the FIR filter.

    Returns:
    - filtered_signal (np.ndarray): Filtered signal.
    """
    filtered_signal = np.column_stack((
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 0]),  # Filter I
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 1])   # Filter Q
    ))

    return filtered_signal

def apply_noise(signal: np.ndarray, noise_level: float=None) -> Tuple[np.ndarray, float]:
    """
    Apply noise to the signal.
    
    Parameters:
    - signal (np.ndarray): Input signal.
    - noise_level (float): Noise level. If None, a random noise level is chosen.

    Returns:
    - noisy_signal (np.ndarray): Noisy signal.
    - noise_level (float): Noise level used.
    """
    if noise_level is None:
        noise_level = np.random.choice(noise_levels)

    noise_range = noise_level * np.max(np.abs(signal))

    noise = np.random.normal(0, noise_range, signal.shape)
    noisy_signal = signal + noise

    return noisy_signal, noise_level

def generate_data(dataset_size: int=DATASET_SIZE, ratio: float=0.5, plot=False, store=False) -> None:
    """
    Generate the data for PulseMatch mimicking the data flow of digital communication systems and store
    it in an HDF5 file.
    
    Parameters:
    - dataset_size (int): Number of samples to generate.
    - ratio (float): Ratio of known to random filters.
    - plot (bool): If True, plot the generated signals and filters.
    - store (bool): If True, store the generated data in HDF5 format.

    Returns:
    - None
    """
    for _ in range(dataset_size):
        # Generate a random bitstream
        bitstream, num_bits = generate_bitstream()

        # Apply modulation
        modulated_signal, modulation_type = apply_modulation(bitstream)

        # Upsample and apply the pulse shaping filter
        if np.random.rand() < ratio:
            filter_taps, filter_name = generate_known_filter(modulation_type)
        else:
            modulated_signal, sps = apply_upsampling(modulated_signal)
            filter_taps, filter_name = generate_random_filter(sps)

        # Apply the FIR filter
        filtered_signal = apply_filter(modulated_signal, filter_taps)

        # Apply noise
        generated_signal, noise_level = apply_noise(filtered_signal)

        if plot:
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            plt.subplot(4, 1, 1)
            plt.plot(filtered_signal, label='Pre-Noise Signal')
            plt.title('Pre-Noise Signal')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(generated_signal[:, 0], label='I Component')
            plt.title('I Component')
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(generated_signal[:, 1], label='Q Component', color='orange')
            plt.title('Q Component')
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.stem(filter_taps, label='FIR Filter Taps')
            plt.title('FIR Filter Taps')
            plt.grid(True)
            plt.suptitle(f'Modulation: {modulation_type}, Filter: {filter_name}, Noise Level: {noise_level:.2f}, Bitstream Size: {num_bits}')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # TODO: Store the generated data in HDF5 format
        if store:
            pass

# View the generated data
if __name__ == "__main__":
    generate_data(dataset_size=10, ratio=0.5, plot=True, store=False)