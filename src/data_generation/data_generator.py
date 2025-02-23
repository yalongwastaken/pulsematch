# Author: Anthony Yalong
# Description: This file generates the data for PulseMatch.

# TODO: Add windowing function to the generated FIR filter taps to make it mimic practical pulse shaping filters.
# TODO: Add symbol rates to the generated FIR filter taps to make it mimic practical pulse shaping filters.
# TODO: Add normalization to the generated FIR filter taps to make it mimic practical pulse shaping filters.

# Imports
import modulation
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple

DATASET_SIZE = 1000000

# NOTE: BITSTREAM SOURCES:
#   The bitstream range of 1024 to 65536 is supported by typical frame sizes in various digital communication 
#   systems. For instance, Ethernet and Wi-Fi frame sizes range from 512 bits (64 bytes) to around 12,000 bits 
#   (1,500 bytes). In LTE and 5G NR, transport blocks vary from 256 bits to over 8,000 bits, depending on channel 
#   conditions and resource allocation. OFDM-based systems typically involve frame sizes from a few hundred bits 
#   to several thousand bits, depending on symbol length and modulation. Additionally, satellite communications 
#   and digital TV systems often operate with frame sizes between 1,000 and 100,000 bits. This range of 1024 to 
#   65536 bits ensures flexibility, accommodating both simpler and more complex transmissions found across different 
#   modern communication networks.
BITSTREAM_SIZE_MIN = 1024
BITSTREAM_SIZE_MAX = 65536

# NOTE: MODULATION SOURCES:
#   ASK modulation/background: https://www.elprocus.com/amplitude-shift-keying-ask-working-and-applications/
#   PSK modulation/background: https://www.elprocus.com/phase-shift-keying-psk-types-and-its-applications/
#   FSK modulation/background: https://eureka.patsnap.com/blog/what-is-fsk/
#   QAM modulation/background: https://eureka.patsnap.com/blog/what-is-qam/ 
modulation_types = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "32QAM",
    "64QAM",
]

# NOTE: UPSAMPLING SOURCES:
#   Upsampling background: https://www.ni.com/docs/en-US/bundle/rfmx-demod/page/samples-per-symbol.html?srsltid=AfmBOorsVA4-lHHO6izwS6Y4FIlAevAkJU2SLIKva6LmogLAY7yzl2wZ
sps_rates = [
    2,
    4,
    8,
    16,
    32,
]

# NOTE: FIR FILTERS SOURCES:
#   FIR filter background: https://thesai.org/Downloads/Volume2No3/Paper%2012-%20Pulse%20Shape%20Filtering%20in%20Wireless%20Communication-A%20Critical%20Analysis.pdf
known_filters = [
    "RRC",
    "RC",
    "Gaussian",
    "Sinc",
]

# NOTE: NOISE SOURCES:
#   AWGN noise/background: https://wirelesspi.com/additive-white-gaussian-noise-awgn/
noise_levels = [
    0,
    0.001,
    0.01,
    0.1,
    0.2,
    0.5,
    1.0,
]

def generate_bitstream(num_bits: int=None) -> np.ndarray:
    """Generate a random bitstream of given length."""
    if num_bits is None:
        num_bits = np.random.uniform(BITSTREAM_SIZE_MIN, BITSTREAM_SIZE_MAX)
        num_bits = int(num_bits)
    
    return np.random.randint(0, 2, num_bits)

def apply_modulation(bitstream: np.ndarray, modulation_type: str=None) -> Tuple[np.ndarray, str]:
    """Apply modulation to the bitstream."""
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

def upsample_signal(signal: np.ndarray, sps: int=None) -> Tuple[np.ndarray, int]:
    """Upsample the signal based on the selected SPS rate."""
    if sps is None:
        sps = np.random.choice(sps_rates)

    upsampled_signal = np.repeat(signal, sps, axis=0)
    return upsampled_signal, sps

# TODO: Add RRC and RC filter taps generation.
def generate_fir_filter_taps() -> np.ndarray:
    """Generate FIR filter taps arbitrarily for pulse shaping."""
    # Randomly select the number of taps (mean=128, std=64)
    num_taps = int(np.random.normal(128, 64))

    # Restrict the number of taps to a range (16, 512)
    num_taps = np.clip(num_taps, 16, 513)

    # Randomly generate commonly used FIR filter taps
    if np.random.rand() < 0.20:
        known_filter = np.random.choice(known_filters)
        if known_filter == "RRC":
            pass
        elif known_filter == "RC":
            pass
        elif known_filter == "Gaussian":
            pass
        elif known_filter == "Sinc":
            pass

    # Generate random FIR filter taps
    filter_taps = np.random.uniform(-1, 1, num_taps)
    return filter_taps

def apply_fir_filter(signal: np.ndarray, filter_taps: np.ndarray=None) -> np.ndarray:
    """Apply FIR filter to the signal."""
    if filter_taps is None:
        filter_taps = generate_fir_filter_taps()

    filtered_signal = np.column_stack((
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 0]),  # Filter I
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 1])   # Filter Q
    ))

    return filtered_signal, filter_taps
   
def apply_noise(signal: np.ndarray, noise_level: float=None) -> Tuple[np.ndarray, float]:
    """Apply noise to the signal."""
    if noise_level is None:
        noise_level = np.random.choice(noise_levels)

    noise = np.random.normal(0, noise_level, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, noise_level

# TODO: Implement HDF5 file generation and storage.
def generate_data(dataset_size: int=DATASET_SIZE, plot=False, store=False) -> None:
    """Generate the data for PulseMatch."""
    for _ in range(dataset_size):
        # Generate a random bitstream
        bitstream = generate_bitstream()

        # Apply modulation
        modulated_signal, modulation_type = apply_modulation(bitstream)

        # Upsample the signal
        upsampled_signal, sps_rate = upsample_signal(modulated_signal)

        # Apply the pulse shaping filter
        filtered_signal, filter_taps = apply_fir_filter(upsampled_signal)

        # Apply noise
        generated_signal, noise_level = apply_noise(filtered_signal)

        if plot:
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            """ 
            plt.subplot(4, 1, 1)
            plt.plot(filtered_signal, label='Pre-noise Signal')
            plt.title('Pre-noise Signal')
            plt.grid(True)
            """
            plt.subplot(4, 1, 1)
            plt.plot(upsampled_signal, label='Pre-Filtered Signal')
            plt.title('Pre-Filtered Signal')
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

            plt.suptitle(f'Signal Data: {modulation_type}, SPS: {sps_rate}, Noise Level: {noise_level}')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # TODO: Store the generated data in HDF5 format
        if store:
            pass

if __name__ == "__main__":
    generate_data(dataset_size=10, plot=True, store=False)