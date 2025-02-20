# Author: Anthony Yalong
# Description: Script file to apply different, specified modulation schemes to an input bitstream.

# Import
import numpy as np
from typing import Tuple

# BPSK modulation
def modulate_bpsk(bitstream: np.ndarray) -> np.ndarray:
    """Apply BPSK modulation to the input signal."""
    # BPSK mapping IQ: 
    #   0 -> [-1,0] 
    #   1 -> [+1,0]
    modulated_signal_i = 2 * bitstream - 1
    modulated_signal_q = np.zeros(len(bitstream))
    modulated_signal = np.stack((modulated_signal_i, modulated_signal_q), axis=-1)
    return modulated_signal

# QPSK modulation
def modulate_qpsk(bitstream: np.ndarray) -> np.ndarray:
    """Apply QPSK modulation to the input signal."""
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
    
    return modulated_signal

# 8PSK modulation
def modulate_8psk(bitstream: np.ndarray) -> np.ndarray:
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

    return modulated_signal

def modulate_16qam(bitstream: np.ndarray) -> np.ndarray:
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

    return modulated_signal

def modulate_32qam(bitstream: np.ndarray) -> np.ndarray:
    pass

def modulate_64qam(bitstream: np.ndarray) -> np.ndarray:
    pass

def modulate_2fsk(bitstream: np.ndarray) -> np.ndarray:
    pass

def modulate_4fsk(bitstream: np.ndarray) -> np.ndarray:
    pass