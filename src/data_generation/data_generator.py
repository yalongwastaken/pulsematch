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
# 3. Upsampling: The modulated signal is upsampled by a factor of SPS (samples per symbol), which simulates the discretization 
#    process of real-world signals. The upsampling process repeats each symbol over multiple samples, which corresponds to the 
#    concept of increasing the sampling rate to capture more precise information about each symbol. This process is relatively
#    simple. To potentially improve the upsampling process, interpolation methods (e.g., linear, cubic) could be used to create 
#    to more accurately mimic real-world signals.
#
# 4. Pulse Shaping: A finite impulse response (FIR) filter is applied to the upsampled signal to introduce pulse shaping. The 
#    FIR filter generation is done arbitrarily, with the random creation of completely random taps and potentially known pulse 
#    shaping filters (e.g., RRC, RC, Gaussian, Sinc). The logic behind FIR Filter generation is as follows:
#
#    - The number of taps for the FIR filter is an integer multiple of the samples per symbol (SPS) to align with the symbol period.
#    - Randomly, either a known filter (e.g., RRC, RC, Gaussian, Sinc) is selected or a completely random FIR filter is generated to 
#      balance the dataset between known and unknown filters.
#    - Random windowing functions (e.g., Hamming, Hanning, etc.) are applied to the generated taps to simulate practical filter 
#      characteristics. Windowing smooths the filter’s impulse response, reducing side lobes and controlling spectral leakage.
#    - Finally, normalization is applied to the filter taps to ensure that the filter has a unit energy, making it similar to 
#      real-world pulse shaping filters, which ensures proper signal power and avoids distortion.
#
# 5. Noise Application: Noise, modeled as additive white Gaussian noise (AWGN), is introduced to simulate real-world channel impairments. 
#    The noise level can be adjusted to represent different signal-to-noise ratios (SNRs). The addition of AWGN, similary to the sps process
#    is relatively simple. To potentially improve the noise application process, more complex noise models (e.g., Rayleigh, Rician)
#
# Notes:
# - Time: 
#   - The code does not explicitly incorporate a time axis. Instead, the data is represented purely in 
#     IQ format, with each symbol mapped to a set number of samples (SPS). This reflects the typical signal representation in digital 
#     communication systems, where signals are sampled at discrete time intervals without a direct time reference.
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
#     concern is addressed, as the symbol rate and sampling rates are implicitly accounted for.
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
#   Windowing function background: TODO
#   AWGN noise/background: https://wirelesspi.com/additive-white-gaussian-noise-awgn/

# TODO: Add RRC and RC filter taps generation.
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

def generate_fir_filter_taps(sps: int) -> np.ndarray:
    """Generate FIR filter taps arbitrarily for pulse shaping."""
    # Define multipliers
    multipliers = np.arange(4, 10, 1)

    # Determine the number of taps based on the symbol rate (SPS)
    num_taps = sps * np.random.choice(multipliers)

    # Restrict the number of taps to a range (16, 512)
    num_taps = np.clip(num_taps, 16, 513)

    # Randomly generate commonly used FIR filter taps
    if np.random.rand() < 0:
        known_filter = np.random.choice(known_filters)
        if known_filter == "RRC":
            pass
            return common_filters.rrc(num_taps), None, "RRC"
        elif known_filter == "RC":
            pass
            return common_filters.rc(num_taps), None, "RC"
        elif known_filter == "Gaussian":
            pass
            return common_filters.gaussian(num_taps), None, "GAUSSIAN"
        elif known_filter == "Sinc":
            pass
            return common_filters.sinc(num_taps), None, "SINC"
    else:
        # Generate random FIR filter taps
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

    return filter_taps, window_type, None

def apply_fir_filter(signal: np.ndarray, sps: int=None, filter_taps: np.ndarray=None) -> np.ndarray:
    """Apply FIR filter to the signal."""
    if sps is None:
        sps = np.random.choice(sps_rates)

    upsampled_signal = np.repeat(signal, sps, axis=0)
    
    if filter_taps is None:
        filter_taps, windowing_type, known_filter = generate_fir_filter_taps(sps)

    filtered_signal = np.column_stack((
        scipy.signal.lfilter(filter_taps, 1.0, upsampled_signal[:, 0]),  # Filter I
        scipy.signal.lfilter(filter_taps, 1.0, upsampled_signal[:, 1])   # Filter Q
    ))

    return filtered_signal, filter_taps, sps, windowing_type, known_filter
   
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
        filtered_signal, filter_taps, sps_rate, windowing_function, known_filter = apply_fir_filter(modulated_signal)

        # Apply noise
        generated_signal, noise_level = apply_noise(filtered_signal)

        if plot:
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            plt.subplot(4, 1, 1)
            plt.plot(filtered_signal, label='Pre-Noise Signal')
            plt.title('Pre-Filtered Signal')
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

            if windowing_function:
                plt.suptitle(f'Signal Data: {modulation_type} & {num_bits} bits, SPS: {sps_rate},  Windowing: {windowing_function}, Noise Level: {noise_level}')
            else:
                plt.suptitle(f'Signal Data: {modulation_type} & {num_bits} bits, SPS: {sps_rate},  Known Filter: {known_filter}, Noise Level: {noise_level}')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # TODO: Store the generated data in HDF5 format
        if store:
            pass

if __name__ == "__main__":
    generate_data(dataset_size=10, plot=True, store=False)