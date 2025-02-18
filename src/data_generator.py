# Author: Anthony Yalong
# Description: This file generates the data for PulseMatch.

# imports
import numpy as np

modulation_types = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
]

sps_rates = [
    2,
    4,
    8,
    16,
]

# TODO: Look into noise levels.
noise_levels = [
]

# TODO: Explore range of bitstream lengths.
def generate_bitstream() -> np.ndarray:
    """
    Generate a random bitstream of given length.
    """
    num_bits = np.random.uniform(100, 10000)
    return np.random.randint(0, 2, num_bits)

# TODO: Implement modulation schemes.
def apply_modulation(bitstream: np.ndarray) -> np.ndarray, str:
    """
    Apply modulation to the bitstream.
    """
    modulation_type = np.random.choice(modulation_types)

    if modulation_type == "BPSK":
    elif modulation_type == "QPSK":
    elif modulation_type == "8PSK":
    elif modulation_type == "16QAM":
    return modulated_signal, modulation_type

def upsample_signal(signal: np.ndarray) -> np.ndarray, int:
    """
    Upsample the signal based on the selected SPS rate.
    """
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
   
# TODO: Implement noise addition.
def apply_noise(signal: np.ndarray) -> np.ndarray, float:
    """
    Apply noise to the signal.
    """
    return noisy_signal, noise_level

# TODO: Implement normalization.
def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize the signal.
    """
    return []

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

    # Generate FIR filter taps
    filter_taps = generate_fir_filter_taps()

    # Apply FIR filter to the signal
    # TODO: Implement FIR filtering

    # Apply noise
    noisy_signal, noise_level = apply_noise(upsampled_signal)

    # Normalize the signal
    normalized_signal = normalize_signal(noisy_signal)

if __name__ == "__main__":
    filter_taps = generate_fir_filter_taps()
    print("Filter taps:", filter_taps)