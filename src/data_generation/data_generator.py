# Author: Anthony Yalong
# Description: This file generates synthetic data for PulseMatch, a model designed to learn and predict pulse shaping functions
#              used in digital communication systems. The process involves the generation of a random bitstream, modulation,
#              upsampling, pulse shaping via FIR filters, and the addition of noise. The generated data is in IQ format (In-phase 
#              and Quadrature), which is commonly used in digital communication systems for representing complex signals.
#
# Rationale:
# The code simulates a simplified signal generation pipeline similar to the transmission chain in digital communication systems.
# It performs the following steps:
#
# 1. Bitstream Generation: A random bitstream is generated to simulate real-world data transmission. The length of the bitstream 
#    varies between 1024 and 65536 bits, corresponding to typical frame sizes used in various communication protocols (e.g., Ethernet, 
#    Wi-Fi, LTE, 5G).
# 
# 2. Modulation: The generated bitstream is modulated using common modulation schemes such as BPSK, QPSK, 8PSK, 16QAM, and higher-order 
#    QAM schemes. These modulations are representative of real-world digital communication systems that map bits to symbols in the 
#    complex plane.
#
# 3. Pulse Shaping: The modulated signal is upsampled and passed through a pulse shaping filter (FIR filter). The filter is either a known
#    filter (e.g., RRC, RC, Gaussian, Sinc) or a randomly generated filter. The purpose of pulse shaping is to control the bandwidth
#    of the transmitted signal and reduce inter-symbol interference (ISI). The generation of the filters follows two approaches, which 
#    align to the theoretical background of digital communication systems:
#
#    - Known Filters: The known filters used to synthesize data are Root Raised Cosine (RRC), Raised Cosine (RC), Gaussian, and Sinc filters.
#      These filters are defined be randomly selecting a bitrate, sampling rate, and roll-of factor, which are used to calculate and determine the
#      filter and its requirements. The filter taps are generated using commpy and normalized to have unit energy, mimicking practical pulse shaping filters.
#      Randomly Generated: Bitrate, Sampling Rate, and Roll-off Factor
#      1. Bitrate: The bitrate is randomly selected from a set of common bitrates (1 kbps to 5 Gbps) to represent different data rates.
#      2. Sampling Rate: The sampling rate is randomly selected from a set of multipliers (2, 4, 8, 16) to represent different oversampling rates.
#      3. Roll-off Factor: The roll-off factor is randomly selected from a range of (0, 1) to represent different filter shapes.
#      # Calculated Requirements: number of taps, symbol rate, symbol period, and roll-off factor.
#      Symbol Rate: The symbol rate is calculated by dividing the bitrate by the modulation type.
#      Symbol Period: The symbol period is determined by taking the reciprocal of the symbol rate.
#      Number of Taps: The number of taps is calculated based on applying a random integer multiple (4, 10) to the SPS. (SPS = symbol rate * symbol period).
#      Roll-off Factor: See above.
#
#    - Randomly Generated: The random filter generation process involves generating a random number of taps based a multiple (4, 10) of the SPS. Next, based
#      on the number of taps, the filter taps are generated using a uniform distribution between -1 and 1. Following this, a random windowing function (Hamming, Hanning,
#      Blackman, Bartlett) is applied to the filter taps, mimicking the windowing process used in practical FIR filter design. Finally, the filter taps are normalized to have unit energy.
#
# 4. Noise Application: Noise, modeled as additive white Gaussian noise (AWGN), is introduced to simulate real-world channel impairments. 
#    The noise level can be adjusted to represent different signal-to-noise ratios (SNRs). The addition of AWGN, similary to the sps process
#    is relatively simple. To potentially improve the noise application process, more complex noise models (e.g., Rayleigh, Rician)
#
# Notes:
# - Time: 
#   - The code does not explicitly incorporate a time axis. Instead, the data is represented purely in IQ format. For the known filters, the time axis
#    is implicitly represented through the filter taps, which are generated based on predefined bitrates and sampling. For the randomly generated filters,
#    the time axis is represented through the number of taps, which are calculated based on predefined SPS rates. Overall, the time axis is not explicitly
#    included in the implementation, as the focus is on generating synthetic data for training a model to learn pulse shaping functions. Note, there is 
#    this discrepency between teh two approaches, as the known filters' requirements for generation are different/more dependent on known system parameters.
#   
# - IQ Representation: 
#   - The IQ format is a standard in digital communications where the signal is represented by two orthogonal 
#     components: the in-phase (I) and quadrature (Q) components. These components carry the encoded information and are used to 
#     represent complex-valued signals.
#
# - Theoretical Alignment:
#   - In theory, digital communication systems rely on a combination of modulation schemes, pulse shaping, and noise modeling 
#     to encode and transmit information efficiently and accurately. This code attempts to accurately reflect that pipeline
#     Through the previously mentioned process. While the time axis is not explicitly included in the implementation, this 
#     concern is addressed, as the filters are generated with the time axis in mind (See above).
#
# - Limitations: 
#   - While this approach mimics the key theoretical components of digital communication, the absence 
#     of an explicit time axis means that the symbol rate is not directly/explicitly represented as a time interval. Instead, the data is sampled 
#     based on the SPS rate. In real systems, time and symbol rate are used to align the transmitted signal with a receiver's 
#     sampling process. However, for training a model to learn pulse shaping functions, I believe this approach is sufficient,
#     as the focus of PulseMatch is to determine if a machine learning model can learn any arbitrary pulse shaping function given
#     different modulation schemes, upsampling rates, and noise levels and not necessarily about receiver alignment.
#
# Sources:
#   ASK modulation/background: https://www.elprocus.com/amplitude-shift-keying-ask-working-and-applications/
#   PSK modulation/background: https://www.elprocus.com/phase-shift-keying-psk-types-and-its-applications/
#   FSK modulation/background: https://eureka.patsnap.com/blog/what-is-fsk/
#   QAM modulation/background: https://eureka.patsnap.com/blog/what-is-qam/ 
#   Upsampling background: https://www.ni.com/docs/en-US/bundle/rfmx-demod/page/samples-per-symbol.html?srsltid=AfmBOorsVA4-lHHO6izwS6Y4FIlAevAkJU2SLIKva6LmogLAY7yzl2wZ
#   FIR filter background: https://thesai.org/Downloads/Volume2No3/Paper%2012-%20Pulse%20Shape%20Filtering%20in%20Wireless%20Communication-A%20Critical%20Analysis.pdf
#   Windowing function background: https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://library.oapen.org/bitstream/20.500.12657/41686/1/9781466515840.pdf&ved=2ahUKEwiG7ZyivN-LAxXwFVkFHUEmAYAQFnoECCoQAQ&usg=AOvVaw1fb9PgMOw9l5kNdrzzhSNF 
#   AWGN noise/background: https://wirelesspi.com/additive-white-gaussian-noise-awgn/

# TODO: Implement HDF5 file generation and storage.
# TODO: Clean code.

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

def generate_bitstream(num_bits: int=None) -> np.ndarray:
    """Generate a random bitstream of given length."""
    if num_bits is None:
        num_bits = np.random.uniform(BITSTREAM_SIZE_MIN, BITSTREAM_SIZE_MAX)
        num_bits = int(num_bits)
    
    return np.random.randint(0, 2, num_bits), num_bits

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

def generate_known_filter(modulation_type: str) -> np.ndarray:
    """Generate a known FIR filter (RRC, RC, etc.)."""
    known_filter = np.random.choice(known_filters)
    if known_filter == "RRC":
        return common_filters.rrc(modulation_type), "RRC"
    elif known_filter == "RC":
        return common_filters.rc(modulation_type), "RC"
    elif known_filter == "Gaussian":
        return common_filters.gaussian(modulation_type), "GAUSSIAN"
    elif known_filter == "Sinc":
        return common_filters.sinc(modulation_type), "SINC"

def apply_upsampling(signal: np.ndarray, sps: int=None) -> np.ndarray:
    """Apply upsampling to the modulated signal."""
    # Randomly select a symbol rate (SPS)
    if sps is None:
        sps = np.random.choice(sps_rates)

    # Upsample the signal by a random factor
    signal = np.repeat(signal, sps, axis=0)

    return signal, sps

def generate_random_filter(sps: int=None) -> np.ndarray:
    """Generate a random FIR filter."""
    # Define multipliers
    multipliers = np.arange(4, 10, 1)

    # Determine the number of taps based on the symbol rate (SPS)
    num_taps = sps * np.random.choice(multipliers)

    # Restrict the number of taps to a range (16, 512)
    num_taps = np.clip(num_taps, 16, 513)

    filter_taps = np.random.uniform(-1, 1, num_taps)

    # Apply a random windowing function
    window_type = np.random.choice(windowing_functions)
    if window_type == "hamming":
        window = np.hamming(num_taps)
    elif window_type == "hanning":
        window = np.hanning(num_taps)
    elif window_type == "blackman":
        window = np.blackman(num_taps)
    elif window_type == "bartlett":
        window = np.bartlett(num_taps)

    filter_taps *= window

    # Normalize the filter taps to have unit energy, mimicking practical pulse shaping filters
    filter_taps /= np.linalg.norm(filter_taps, 2)

    return filter_taps

def apply_filter(signal: np.ndarray, filter_taps: np.ndarray) -> np.ndarray:
    """Apply the FIR filter to the modulated signal."""
    filtered_signal = np.column_stack((
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 0]),  # Filter I
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 1])   # Filter Q
    ))

    return filtered_signal

def apply_noise(signal: np.ndarray, noise_level: float=None) -> Tuple[np.ndarray, float]:
    """Apply noise to the signal."""
    if noise_level is None:
        noise_level = np.random.choice(noise_levels)

    noise_range = noise_level * np.max(np.abs(signal))

    noise = np.random.normal(0, noise_range, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, noise_level

def generate_data(dataset_size: int=DATASET_SIZE, plot=False, store=False) -> None:
    """Generate the data for PulseMatch."""
    for _ in range(dataset_size):
        # Generate a random bitstream
        bitstream, num_bits = generate_bitstream()

        # Apply modulation
        modulated_signal, modulation_type = apply_modulation(bitstream)

        # Upsample and apply the pulse shaping filter
        if np.random.rand() < 0:
            filter_taps, known_filter_name = generate_known_filter(modulation_type)
            print(f"Filter Taps Shape: {filter_taps.shape}")
        else:
            modulated_signal, sps = apply_upsampling(modulated_signal)
            filter_taps = generate_random_filter(sps)

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

            """ 
            plt.subplot(4, 1, 1)
            plt.plot(modulated_signal, label='Modulated Signal')
            plt.title('Pre-Filtered Signal')
            plt.grid(True)
            """

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
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # TODO: Store the generated data in HDF5 format
        if store:
            pass

if __name__ == "__main__":
    generate_data(dataset_size=10, plot=True, store=False)