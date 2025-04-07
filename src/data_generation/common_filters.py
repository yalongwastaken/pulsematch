# Author: Anthony Yalong
# Description: Script file to generate common filters (RRC, RC, SINC, GAUSSIAN) for PulseMatch.

# Imports
import numpy as np
import commpy as cp
import matplotlib.pyplot as plt

from typing import Tuple

bitrates = [
    1000,       # 1 kbps
    2000,       # 2 kbps
    5000,       # 5 kbps
    10000,      # 10 kbps
    20000,      # 20 kbps
    50000,      # 50 kbps
    100000,     # 100 kbps
    200000,     # 200 kbps
    500000,     # 500 kbps
    1000000,    # 1 Mbps
    2000000,    # 2 Mbps
    5000000,    # 5 Mbps
    10000000,   # 10 Mbps
    20000000,   # 20 Mbps
    50000000,   # 50 Mbps
    100000000,  # 100 Mbps
    200000000,  # 200 Mbps
    500000000,  # 500 Mbps
    1000000000, # 1 Gbps
    2000000000, # 2 Gbps
    5000000000, # 5 Gbps
]

sampling_rate_multiplier = [
    2,  # 2x Nyquist
    4,  # 4x Nyquist
    8,  # 8x Nyquist
    16, # 16x Nyquist
]

# Modulation types
modulation_types = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
    "32QAM": 5,
    "64QAM": 6,
}

def rrc(modulation_type: str=None, num_taps: int=None) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a root raised cosine filter.
    
    Parameters
    ----------
    - modulation_type (str): The modulation type (e.g., "BPSK", "QPSK", etc.)

    Returns
    -------
    - rrc_filter (np.ndarray): The generated root raised cosine filter.
    - symbol_rate (float): The symbol rate.
    - sampling_rate (float): The sampling rate.
    - roll_off (float): The roll-off factor.
    """
    # Ranom sampling rate and bitrate
    bitrate = np.random.choice(bitrates)
    sampling_rate = np.random.choice(sampling_rate_multiplier) * bitrate

    # Determine symbol rate
    symbol_rate = bitrate / modulation_types[modulation_type]

    # Determine symbol period
    symbol_period = 1 / symbol_rate

    # Random roll-off factor
    roll_off = np.random.uniform(0, 1)

    # Number of taps
    if num_taps is None:
        num_taps = int(sampling_rate * symbol_period) * np.random.randint(4, 10)

    # Generate the RRC filter using commpy
    rrc_filter = cp.rrcosfilter(num_taps, roll_off, symbol_period, sampling_rate)[1] 

    # Normalize the filter energy to 1
    rrc_filter /= np.sqrt(np.sum(rrc_filter**2))

    return rrc_filter, bitrate, sampling_rate, roll_off

def rc(modulation_type: str=None, num_taps: int=None) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a raised cosine filter.
    
    Parameters
    ----------
    - modulation_type (str): The modulation type (e.g., "BPSK", "QPSK", etc.)

    Returns
    -------
    - rc_filter (np.ndarray): The generated raised cosine filter.
    - symbol_rate (float): The symbol rate.
    - sampling_rate (float): The sampling rate.
    - roll_off (float): The roll-off factor.
    """
    # Ranom sampling rate and bitrate
    bitrate = np.random.choice(bitrates)
    sampling_rate = np.random.choice(sampling_rate_multiplier) * bitrate

    # Determine symbol rate
    symbol_rate = bitrate / modulation_types[modulation_type]

    # Determine symbol period
    symbol_period = 1 / symbol_rate

    # Random roll-off factor
    roll_off = np.random.uniform(0, 1)

    # Number of taps
    if num_taps is None:
        num_taps = int(sampling_rate * symbol_period) * np.random.randint(4, 10)

    # Generate the RC filter using commpy
    rc_filter = cp.rcosfilter(num_taps, roll_off, symbol_period, sampling_rate)[1]

    # Normalize the filter energy to 1
    rc_filter /= np.sqrt(np.sum(rc_filter**2))

    return rc_filter, bitrate, sampling_rate, roll_off

def gaussian(modulation_type: str=None, num_taps: int=None) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a Gaussian filter.
    
    Parameters
    ----------
    - modulation_type (str): The modulation type (e.g., "BPSK", "QPSK", etc.)

    Returns
    -------
    - gaussian_filter (np.ndarray): The generated Gaussian filter.
    - symbol_rate (float): The symbol rate.
    - sampling_rate (float): The sampling rate.
    - roll_off (float): The roll-off factor.
    """
    # Ranom sampling rate and bitrate
    bitrate = np.random.choice(bitrates)
    sampling_rate = np.random.choice(sampling_rate_multiplier) * bitrate

    # Determine symbol rate
    symbol_rate = bitrate / modulation_types[modulation_type]

    # Determine symbol period
    symbol_period = 1 / symbol_rate

    # Random roll-off factor
    roll_off = np.random.uniform(0, 1)

    # Number of taps
    if num_taps is None:
        num_taps = int(sampling_rate * symbol_period) * np.random.randint(4, 10)

    # Generate the Gaussian filter using commpy
    gaussian_filter = cp.gaussianfilter(num_taps, roll_off, symbol_period, sampling_rate)[1]

    # Normalize the filter energy to 1
    gaussian_filter /= np.sqrt(np.sum(gaussian_filter**2))

    return gaussian_filter, bitrate, sampling_rate, roll_off

def sinc(modulation_type: str=None, num_taps: int=None) -> np.ndarray:
    """
    Generate a sinc filter.
    
    Parameters
    ----------
    - modulation_type (str): The modulation type (e.g., "BPSK", "QPSK", etc.)

    Returns
    -------
    - sinc_filter (np.ndarray): The generated sinc filter.
    - symbol_rate (float): The symbol rate.
    - sampling_rate (float): The sampling rate.
    - roll_off (float): The roll-off factor.
    """
    # Ranom sampling rate and bitrate
    bitrate = np.random.choice(bitrates)
    sampling_rate = np.random.choice(sampling_rate_multiplier) * bitrate

    # Determine symbol rate
    symbol_rate = bitrate / modulation_types[modulation_type]

    # Determine symbol period
    symbol_period = 1 / symbol_rate

    # Random roll-off factor
    roll_off = 0

    # Number of taps
    if num_taps is None:
        num_taps = int(sampling_rate * symbol_period) * np.random.randint(4, 10)

    # Generate the Sinc filter using commpy
    sinc_filter = cp.rcosfilter(num_taps, roll_off, symbol_period, sampling_rate)[1]

    # Normalize the filter energy to 1
    sinc_filter /= np.sqrt(np.sum(sinc_filter**2))

    return sinc_filter, bitrate, sampling_rate, roll_off


if __name__ == "__main__":
    # Test RRC filter
    rrc_filter = rrc("BPSK")
    plt.plot(rrc_filter)
    plt.title("Root Raised Cosine Filter")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Test RC filter
    rc_filter = rc("BPSK")
    plt.plot(rc_filter)
    plt.title("Raised Cosine Filter")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Test Gaussian filter
    gaussian_filter = gaussian("BPSK")
    plt.plot(gaussian_filter)
    plt.title("Gaussian Filter")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Test Sinc filter
    sinc_filter = sinc("BPSK")
    plt.plot(sinc_filter)
    plt.title("Sinc Filter")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()