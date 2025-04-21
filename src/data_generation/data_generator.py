# Author: Anthony Yalong
# Description: This file generates synthetic data for PulseMatch, a model designed to learn and predict pulse shaping functions
#              used in digital communication systems. The process involves the generation of a random bitstream, modulation,
#              upsampling, pulse shaping via FIR filters, and the addition of noise. The generated data is in IQ format (In-phase 
#              and Quadrature), which is commonly used in digital communication systems for representing complex signals.

# Imports
import h5py
import modulation
import common_filters
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple

DEFAULT_FILEPATH = "src/data/dataset.h5"
DATASET_SIZE = 1000000
BITSTREAM_SIZE_MIN = 1024
BITSTREAM_SIZE_MAX = 65536
NUM_FIR_TAPS = None

modulation_types = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "32QAM",
    "64QAM",
]

modulation_types_dict = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
    "32QAM": 5,
    "64QAM": 6,
}

sampling_rate_multiplier = [
    2,  # 2x Nyquist
    4,  # 4x Nyquist
    8,  # 8x Nyquist
]

known_filters = [
    "Rectangular",
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
    
    Parameters
    ----------
    - num_bits (int): Length of the bitstream. If None, a random length between BITSTREAM_SIZE_MIN and BITSTREAM_SIZE_MAX is chosen.

    Returns
    ----------
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

    Parameters
    ----------
    - bitstream (np.ndarray): Input bitstream.
    - modulation_type (str): Type of modulation to apply. If None, a random modulation type is chosen.

    Returns
    ----------
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

def generate_known_filter(modulation_type: str, known_filter: str=None, num_taps: int=NUM_FIR_TAPS) -> Tuple[np.ndarray, str]:
    """
    Generate a known FIR filter (RRC, RC, etc.).
    
    Parameters
    ----------
    - modulation_type (str): Type of modulation used.
    - known_filter (str): Type of known filter to generate. If None, a random known filter is chosen.

    Returns
    ----------
    - filter_taps (np.ndarray): Coefficients of the FIR filter.
    - filter_name (str): Name of the filter.
    """
    if known_filter is None:
        known_filter = np.random.choice(known_filters)

    # Root Raised Cosine (RRC) filter
    if known_filter == "RRC":
        rrc_filter, symbol_rate, sampling_rate, roll_off = common_filters.rrc(modulation_type, num_taps)
        return rrc_filter, symbol_rate, sampling_rate, roll_off, "RRC"
    
    # Raised Cosine (RC) filter
    elif known_filter == "RC":
        rc_filter, symbol_rate, sampling_rate, roll_off = common_filters.rc(modulation_type, num_taps)
        return rc_filter, symbol_rate, sampling_rate, roll_off, "RC"
    
    # Gaussian filter
    elif known_filter == "Gaussian":
        gaussian_filter, symbol_rate, sampling_rate, roll_off = common_filters.gaussian(modulation_type, num_taps)
        return gaussian_filter, symbol_rate, sampling_rate, roll_off, "GAUSSIAN"
    
    # Sinc filter
    elif known_filter == "Sinc":
        sinc_filter, symbol_rate, sampling_rate, roll_off = common_filters.sinc(modulation_type, num_taps)
        return sinc_filter, symbol_rate, sampling_rate, roll_off, "SINC"
    
    # Rectangular filter
    elif known_filter == "Rectangular":
        rectangular_filter, symbol_rate, sampling_rate, roll_off = common_filters.rectangular(modulation_type, num_taps)
        return rectangular_filter, symbol_rate, sampling_rate, roll_off, "RECTANGULAR"


def generate_random_filter(sps: int=None, modulation_type: str=None, window_type: str=None, num_taps: int=NUM_FIR_TAPS) -> Tuple[np.ndarray, str]:
    """
    Generate a random FIR filter.
    
    Parameters
    ----------
    - sps (int): Samples per symbol.
    - window_type (str): Type of windowing function. If None, a random windowing function is chosen.

    Returns
    ----------
    - filter_taps (np.ndarray): Coefficients of the FIR filter.
    - filter_name (str): Name of the filter.
    """
    # Determine sps based on modulation type and sampling rate multipler
    if sps is None:
        sps = np.random.choice(sampling_rate_multiplier) * modulation_types_dict[modulation_type]

    # Determine the number of taps based on sps and modulation type
    if num_taps is None:
        num_taps = sps * np.random.randint(4, 10)

    filter_taps = np.random.uniform(-1, 1, num_taps)

    # Apply a random windowing function
    if window_type is None:
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

    return filter_taps, window_type, "RANDOM"

def apply_filter(signal: np.ndarray, filter_taps: np.ndarray) -> np.ndarray:
    """
    Apply the FIR filter to the modulated signal.
    
    Parameters
    ----------
    - signal (np.ndarray): Input modulated signal.
    - filter_taps (np.ndarray): Coefficients of the FIR filter.

    Returns
    ----------
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
    
    Parameters
    ----------
    - signal (np.ndarray): Input signal.
    - noise_level (float): Noise level. If None, a random noise level is chosen.

    Returns
    ----------
    - noisy_signal (np.ndarray): Noisy signal.
    - noise_level (float): Noise level used.
    """
    if noise_level is None:
        noise_level = np.random.choice(noise_levels)

    noise_range = noise_level * np.max(np.abs(signal))

    noise = np.random.normal(0, noise_range, signal.shape)
    noisy_signal = signal + noise

    return noisy_signal, noise_level

def generate_data(
        dataset_size: int=DATASET_SIZE,
        num_bits: int=None,
        modulation_type: str=None,
        ratio: float=0.25, 
        known_filter: str=None,
        sps: int=None,
        window_type: str=None,
        noise_level: float=None,
        plot=False, 
        store=False,
        verbose=True,
        write_mode='w',
        filepath: str=DEFAULT_FILEPATH,
        ) -> None:
    """
    Generate the data for PulseMatch mimicking the data flow of digital communication systems and store it.
    
    Parameters
    ----------
    - dataset_size (int): Number of samples to generate.
    - num_bits (int): Length of the bitstream. If None, a random length is chosen.
    - modulation_type (str): Type of modulation to apply. If None, a random modulation type is chosen.
    - ratio (float): Ratio of known filters to random filters.
    - known_filter (str): Type of known filter to generate. If None, a random known filter is chosen.
    - sps (int): Symbol rate. If None, a random symbol rate is chosen.
    - window_type (str): Type of windowing function. If None, a random windowing function is chosen.
    - noise_level (float): Noise level. If None, a random noise level is chosen.
    - plot (bool): Whether to plot the generated data.
    - store (bool): Whether to store the generated data in HDF5 format.
    - verbose (bool): Whether to print the signal characteristics.
    - write_mode (str): Write mode for storing data in HDF5 format.
    - filepath (str): Path to store the generated data.

    Returns
    ----------
    - None
    """
    if store:
        # I/O
        all_signals = []
        all_filter_taps = []

        # Shared characterstics
        all_bitstream_sizes = []
        all_modulation_types = []
        all_noise_levels = []
        all_filter_names = []

        # Known filter characteristics
        all_bitrates = []
        all_sampling_rates = []
        all_roll_offs = []

        # Random filter characteristics
        all_window_types = []

    for _ in range(dataset_size):
        # Generate a random bitstream
        _bitstream, _num_bits = generate_bitstream(num_bits=num_bits)

        # Apply modulation
        _modulated_signal, _modulation_type = apply_modulation(bitstream=_bitstream, modulation_type=modulation_type)

        # apply the pulse shaping filter
        if np.random.rand() < ratio:
            _filter_taps, _bitrate, _sampling_rate, _roll_off, _filter_name = generate_known_filter(modulation_type=_modulation_type, known_filter=known_filter)
        else:
            _filter_taps, _window_type, _filter_name = generate_random_filter(sps=sps, modulation_type=_modulation_type, window_type=window_type)

        _filtered_signal = apply_filter(signal=_modulated_signal, filter_taps=_filter_taps)

        # Apply noise
        _signal, _noise_level = apply_noise(signal=_filtered_signal, noise_level=noise_level)

        # Print signal characteristics
        if verbose:
            print(f"Bitstream Size: {_num_bits}, Modulation: {_modulation_type}, Filter Type: {_filter_name}, Noise Level: {_noise_level:.2f}, Num Taps: {len(_filter_taps)}")

            if _filter_name == "RANDOM":
                print(f"Window Type: {_window_type}")

            else:
                print(f"Bit Rate: {_bitrate}, Sampling Rate: {_sampling_rate}, Roll-off Factor: {_roll_off}")

        if plot:
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            plt.subplot(4, 1, 1)
            plt.plot(_bitstream, label='Bitsream')
            plt.title('Bitstream')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(_signal[:, 0], label='I Component')
            plt.title('I Component')
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(_signal[:, 1], label='Q Component', color='orange')
            plt.title('Q Component')
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.stem(_filter_taps, label='FIR Filter Taps')
            plt.title('FIR Filter Taps')
            plt.grid(True)
            plt.suptitle(f'Modulation: {_modulation_type}, Filter: {_filter_name}, Noise Level: {_noise_level:.2f}, Bitstream Size: {_num_bits}, Num Taps: {len(_filter_taps)}')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        if store:
            # Store characteristics
            all_signals.append(_signal.flatten())
            all_filter_taps.append(_filter_taps)
            all_bitstream_sizes.append(_num_bits)
            all_modulation_types.append(_modulation_type)
            all_noise_levels.append(_noise_level)
            all_filter_names.append(_filter_name)

            if _filter_name == "RANDOM":
                # Random filter characteristics
                all_window_types.append(_window_type)

                # Known filter characteristics (None)
                all_bitrates.append(-1)
                all_sampling_rates.append(-1)
                all_roll_offs.append(-1)
            else:
                # Random filter characteristics (None)
                all_window_types.append("none")

                # Known filter characteristics
                all_bitrates.append(_bitrate)
                all_sampling_rates.append(_sampling_rate)
                all_roll_offs.append(_roll_off)

    # Store data in tf.data.Dataset       
    if store:
        # Pad FIR filter taps to the maximum length
        max_taps = 432
        all_filter_taps = [np.pad(taps, ((max_taps - len(taps)) // 2, (max_taps - len(taps)) - (max_taps - len(taps)) // 2), mode='constant') for taps in all_filter_taps]

        with h5py.File(filepath, write_mode) as f:
            # Define custom data types
            vlen_arr_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
            string_dtype = h5py.special_dtype(vlen=str)  

            # Input: variable-length IQ signal
            f.create_dataset('signals', (len(all_signals),), dtype=vlen_arr_dtype, data=all_signals)

            # Output: same-length FIR filter taps
            f.create_dataset('filter_taps', data=np.array(all_filter_taps, dtype=np.float32))

            # Shared Auxilary characteristics
            f.create_dataset("bitstream_sizes", data=np.array(all_bitstream_sizes), dtype='int32')
            f.create_dataset("modulation_types", data=all_modulation_types, dtype=string_dtype)
            f.create_dataset("noise_levels", data=np.array(all_noise_levels), dtype='float32')
            f.create_dataset("filter_names", data=all_filter_names, dtype=string_dtype)

            # Known filter Auxilary characteristics
            f.create_dataset("bitrates", data=np.array(all_bitrates), dtype='int32')
            f.create_dataset("sampling_rates", data=np.array(all_sampling_rates), dtype='int32')
            f.create_dataset("roll_offs", data=np.array(all_roll_offs), dtype='float32')

            # Random filter Auxilary characteristics
            f.create_dataset("window_types", data=all_window_types, dtype=string_dtype)

        print(f"Data stored in {filepath}")
        
if __name__ == "__main__":
    # Visualization
    generate_data(
        dataset_size=15000,
        num_bits=None,
        modulation_type=None,
        ratio=1.0, 
        known_filter=None,
        sps=None,
        window_type=None,
        noise_level=None,
        plot=False,
        store=True,
        verbose=False,
        write_mode='w',
        filepath="src/data/dataset_8c.h5",
    )