# Author: Anthony Yalong
# Description: This file generates the data for PulseMatch.

# NOTE: Looking to generate a dataset with approximately 1 million samples.

# TODO: Add reasoning behind the data generation process.
#   1. Source for why the bitstream range is between 100 and 10000.
#   2. Source for why the modulation types are BPSK, QPSK, 8PSK, and 16QAM.
#   3. Source for why the noise levels are between 0.01 and 1.0.
#   4. Source for why the range of FIR filter taaps is between 4 and 513.
#   5. Source for why use a normal distribution for FIR filter taps has a mean of 64 and std of 32. 
#   6. Source for why the upsampling rates are between 2 and 16.
#   7. Source for why the dataset size is between 100000 and 1000000.

# imports
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

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

def apply_modulation(bitstream: np.ndarray, modulation_type: str=None) -> Tuple[np.ndarray, str]:
    """
    Apply modulation to the bitstream.
    """
    if modulation_type is None:
        modulation_type = np.random.choice(modulation_types)

    # BPSK modulation
    if modulation_type == "BPSK":
        # BPSK mapping IQ: 
        #   0 -> [-1,0] 
        #   1 -> [+1,0]
        modulated_signal_i = 2 * bitstream - 1
        modulated_signal_q = np.zeros(len(bitstream))
        modulated_signal = np.stack((modulated_signal_i, modulated_signal_q), axis=-1)
    
    elif modulation_type == "QPSK":
        # QPSK mapping IQ:
        #   00 -> [-1,-1]
        #   01 -> [-1,+1]
        #   10 -> [+1,-1]
        #   11 -> [+1,+1]
        
        # Ensure the bitstream length is even
        if len(bitstream) % 2 != 0:
            bitstream = np.append(bitstream, 0)

        # Reshape the bitstream into pairs of bits
        bitstream_pairs = bitstream.reshape(-1, 2)

        # Map the pairs to QPSK symbols
        modulated_signal = np.zeros((bitstream_pairs.shape[0], 2))
        for _, pair in enumerate(bitstream_pairs):
            if np.array_equal(pair, [0, 0]):
                modulated_signal[_, :] = [-1, -1]
            elif np.array_equal(pair, [0, 1]):
                modulated_signal[_, :] = [-1, +1]
            elif np.array_equal(pair, [1, 0]):
                modulated_signal[_, :] = [+1, -1]
            elif np.array_equal(pair, [1, 1]):
                modulated_signal[_, :] = [+1, +1]

    elif modulation_type == "8PSK":
        # 8PSK mapping IQ:
        #   000 -> [-1,-1]
        #   001 -> [-1,+1]
        #   010 -> [+1,-1]
        #   011 -> [+1,+1]
        #   100 -> [0,-1]
        #   101 -> [0,+1]
        #   110 -> [-1,0]
        #   111 -> [+1,0]
        
        # Ensure the bitstream length is a multiple of 3
        if len(bitstream) % 3 != 0:
            bitstream = np.append(bitstream, np.zeros(3 - (len(bitstream) % 3)))
        
        # Reshape the bitstream into triplets of bits
        bitstream_triplets = bitstream.reshape(-1, 3)

        # Map the triplets to 8PSK symbols
        modulated_signal = np.zeros((bitstream_triplets.shape[0], 2))
        for _, triplet in enumerate(bitstream_triplets):
            if np.array_equal(triplet, [0, 0, 0]):
                modulated_signal[_, :] = [-1, -1]
            elif np.array_equal(triplet, [0, 0, 1]):
                modulated_signal[_, :] = [-1, +1]
            elif np.array_equal(triplet, [0, 1, 0]):
                modulated_signal[_, :] = [+1, -1]
            elif np.array_equal(triplet, [0, 1, 1]):  
                modulated_signal[_, :] = [+1, +1]
            elif np.array_equal(triplet, [1, 0, 0]):
                modulated_signal[_, :] = [0, -1]
            elif np.array_equal(triplet, [1, 0, 1]):
                modulated_signal[_, :] = [0, +1]
            elif np.array_equal(triplet, [1, 1, 0]):
                modulated_signal[_, :] = [-1, 0]
            elif np.array_equal(triplet, [1, 1, 1]):
                modulated_signal[_, :] = [+1, 0]

    # TODO: Implement 16QAM
    elif modulation_type == "16QAM":
        # 16QAM mapping IQ:
        #   0000 -> [-3,-3]
        #   0001 -> [-3,-1]
        #   0010 -> [-3,+1]
        #   0011 -> [-3,+3]
        #   0100 -> [-1,-3]
        #   0101 -> [-1,-1]
        #   0110 -> [-1,+1]
        #   0111 -> [-1,+3]
        #   1000 -> [+1,-3]
        #   1001 -> [+1,-1]
        #   1010 -> [+1,+1]
        #   1011 -> [+1,+3]
        #   1100 -> [+3,-3]
        #   1101 -> [+3,-1]
        #   1110 -> [+3,+1]
        #   1111 -> [+3,+3]
        
        # Ensure the bitstream length is a multiple of 4
        if len(bitstream) % 4 != 0:
            bitstream = np.append(bitstream, np.zeros(4 - (len(bitstream) % 4)))

        # Reshape the bitstream into quadruplets of bits
        bitstream_quadruplets = bitstream.reshape(-1, 4)

        # Map the quadruplets to 16QAM symbols
        modulated_signal = np.zeros((bitstream_quadruplets.shape[0], 2))
        for _, quadruplet in enumerate(bitstream_quadruplets):
            if np.array_equal(quadruplet, [0, 0, 0, 0]):
                modulated_signal[_, :] = [-3, -3]
            elif np.array_equal(quadruplet, [0, 0, 0, 1]):
                modulated_signal[_, :] = [-3, -1]
            elif np.array_equal(quadruplet, [0, 0, 1, 0]):
                modulated_signal[_, :] = [-3, +1]
            elif np.array_equal(quadruplet, [0, 0, 1, 1]):
                modulated_signal[_, :] = [-3, +3]
            elif np.array_equal(quadruplet, [0, 1, 0, 0]):
                modulated_signal[_, :] = [-1, -3]
            elif np.array_equal(quadruplet, [0, 1, 0, 1]):
                modulated_signal[_, :] = [-1, -1]
            elif np.array_equal(quadruplet, [0, 1, 1, 0]):
                modulated_signal[_, :] = [-1, +1]
            elif np.array_equal(quadruplet, [0, 1, 1, 1]):
                modulated_signal[_, :] = [-1, +3]
            elif np.array_equal(quadruplet, [1, 0, 0, 0]):
                modulated_signal[_, :] = [+1, -3]
            elif np.array_equal(quadruplet, [1, 0, 0, 1]):
                modulated_signal[_, :] = [+1,-1]
            elif np.array_equal(quadruplet,[1 ,0 ,1 ,0 ]):
                modulated_signal[_, :] = [+1,+1]
            elif np.array_equal(quadruplet,[1 ,0 ,1 ,1 ]):
                modulated_signal[_, :] = [+1,+3]
            elif np.array_equal(quadruplet,[1 ,1 ,0 ,0 ]):
                modulated_signal[_, :] = [+3,-3]
            elif np.array_equal(quadruplet,[1 ,1 ,0 ,1 ]):
                modulated_signal[_, :] = [+3,-1]
            elif np.array_equal(quadruplet,[1 ,1 ,1 ,0 ]):
                modulated_signal[_, :] = [+3,+1]
            elif np.array_equal(quadruplet,[1 ,1 ,1 ,1 ]):
                modulated_signal[_, :] = [+3,+3]

    return modulated_signal, modulation_type

def upsample_signal(signal: np.ndarray, sps: int=None) -> Tuple[np.ndarray, int]:
    """
    Upsample the signal based on the selected SPS rate.
    """
    if sps is None:
        sps = np.random.choice(sps_rates)

    upsampled_signal = np.repeat(signal, sps, axis=0)
    return upsampled_signal, sps

# TODO: Add RRC and RC filter taps generation.
def generate_fir_filter_taps() -> np.ndarray:
    """
    Generate FIR filter taps arbitrarily for pulse shaping.
    """
    num_taps = int(np.random.normal(64, 32))
    num_taps = np.clip(num_taps, 4, 513)

    # Generate random FIR filter taps
    filter_taps = np.random.uniform(-1, 1, num_taps)
    return filter_taps

def apply_fir_filter(signal: np.ndarray, filter_taps: np.ndarray=None) -> np.ndarray:
    """
    Apply FIR filter to the signal.
    """
    if filter_taps is None:
        filter_taps = generate_fir_filter_taps()

    filtered_signal = np.column_stack((
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 0]),  # Filter I
        scipy.signal.lfilter(filter_taps, 1.0, signal[:, 1])   # Filter Q
    ))

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

# TODO: Implement HDF5 file generation and storage.
def generate_data(dataset_size: int=100000, plot=False, store=False) -> None:
    """
    Generate the data for PulseMatch.
    """
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
        noisy_signal, noise_level = apply_noise(filtered_signal)

        # Normalize the signal
        normalized_signal = normalize_signal(noisy_signal)

        if plot:
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            plt.subplot(4, 1, 1)
            plt.plot(bitstream, label='Bitstream')
            plt.title('Bitstream')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(normalized_signal[:, 0], label='I Component')
            plt.title('I Component')
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(normalized_signal[:, 1], label='Q Component', color='orange')
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