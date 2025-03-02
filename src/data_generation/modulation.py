# Author: Anthony Yalong
# Description: Script file to apply different, specified modulation schemes to an input bitstream.

# Import
import numpy as np
from typing import Tuple

# BPSK modulation
def modulate_bpsk(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply BPSK modulation to the input signal.
    
    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
    # BPSK mapping IQ: 
    #   0 -> [-1,0] 
    #   1 -> [+1,0]

    mapping_bpsk = {
        0: (-1, 0), 
        1: (+1, 0)
    }

    modulated_signal = np.zeros((bitstream.shape[0], 2))
    for _, bit in enumerate(bitstream):
        modulated_signal[_, :] = mapping_bpsk[bit]
    return modulated_signal

# QPSK modulation
def modulate_qpsk(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply QPSK modulation to the input signal.
    
    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
    # QPSK mapping IQ:
    #   00 -> [-1,-1]
    #   01 -> [-1,+1]
    #   10 -> [+1,-1]
    #   11 -> [+1,+1]

    mapping_qpsk = {
        (0,0): (-1,-1), 
        (0,1): (-1,+1), 
        (1,0): (+1,-1), 
        (1,1): (+1,+1),
    }
    
    # Ensure the bitstream length is even
    if len(bitstream) % 2 != 0:
        bitstream = np.append(bitstream, 0)

    # Reshape the bitstream into pairs of bits
    bitstream_pairs = bitstream.reshape(-1, 2)

    # Map the pairs to QPSK symbols
    modulated_signal = np.zeros((bitstream_pairs.shape[0], 2))
    for _, pair in enumerate(bitstream_pairs):
        modulated_signal[_, :] = mapping_qpsk[tuple(pair)]
    
    return modulated_signal

# 8PSK modulation
def modulate_8psk(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply 8PSK modulation to the input signal.

    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
    # 8PSK mapping IQ:
    #   000 -> [-1,-1]
    #   001 -> [-1,+1]
    #   010 -> [+1,-1]
    #   011 -> [+1,+1]
    #   100 -> [0,-1]
    #   101 -> [0,+1]
    #   110 -> [-1,0]
    #   111 -> [+1,0]

    mapping_8psk = {
        (0,0,0): (-1,-1), (0,0,1): (-1,+1), (0,1,0): (+1,-1), (0,1,1): (+1,+1),
        (1,0,0): ( 0,-1), (1,0,1): ( 0,+1), (1,1,0): (-1, 0), (1,1,1): (+1, 0),
    }
    
    # Ensure the bitstream length is a multiple of 3
    if len(bitstream) % 3 != 0:
        bitstream = np.append(bitstream, np.zeros(3 - (len(bitstream) % 3)))
    
    # Reshape the bitstream into triplets of bits
    bitstream_triplets = bitstream.reshape(-1, 3)

    # Map the triplets to 8PSK symbols
    modulated_signal = np.zeros((bitstream_triplets.shape[0], 2))
    for _, triplet in enumerate(bitstream_triplets):
        modulated_signal[_, :] = mapping_8psk[tuple(triplet)]

    return modulated_signal

def modulate_16qam(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply 16QAM modulation to the input signal.

    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
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

    mapping_16qam = {
        (0,0,0,0): (-3,-3), (0,0,0,1): (-3,-1), (0,0,1,0): (-3,+1), (0,0,1,1): (-3,+3),
        (0,1,0,0): (-1,-3), (0,1,0,1): (-1,-1), (0,1,1,0): (-1,+1), (0,1,1,1): (-1,+3),
        (1,0,0,0): (+1,-3), (1,0,0,1): (+1,-1), (1,0,1,0): (+1,+1), (1,0,1,1): (+1,+3),
        (1,1,0,0): (+3,-3), (1,1,0,1): (+3,-1), (1,1,1,0): (+3,+1), (1,1,1,1): (+3,+3),
    }
    
    # Ensure the bitstream length is a multiple of 4
    if len(bitstream) % 4 != 0:
        bitstream = np.append(bitstream, np.zeros(4 - (len(bitstream) % 4)))

    # Reshape the bitstream into quadruplets of bits
    bitstream_quadruplets = bitstream.reshape(-1, 4)

    # Map the quadruplets to 16QAM symbols
    modulated_signal = np.zeros((bitstream_quadruplets.shape[0], 2))
    for _, quadruplet in enumerate(bitstream_quadruplets):
        modulated_signal[_, :] = mapping_16qam[tuple(quadruplet)]
    
    return modulated_signal

def modulate_32qam(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply 32QAM modulation to the input signal.

    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
    # 32QAM mapping IQ:
    #   00000 -> [-1,-1]
    #   00001 -> [-3,-1]
    #   00010 -> [-1,-3]
    #   00011 -> [-3,-3]
    #   00100 -> [-1,+1]
    #   00101 -> [-3,+1]
    #   00110 -> [-1,+3]
    #   00111 -> [-3,+3]
    #   01000 -> [+1,-1]
    #   01001 -> [+3,-1]
    #   01010 -> [+1,-3]
    #   01011 -> [+3,-3]
    #   01100 -> [+1,+1]
    #   01101 -> [+3,+1]
    #   01110 -> [+1,+3]
    #   01111 -> [+3,+3]
    #   10000 -> [-3,-5]
    #   10001 -> [-5,-1]
    #   10010 -> [-1,-5]
    #   10011 -> [-5,-3]
    #   10100 -> [-3,+5]
    #   10101 -> [-5,+1]
    #   10110 -> [-1,+5]
    #   10111 -> [-5,+3]
    #   11000 -> [+3,-5]
    #   11001 -> [+5,-1]
    #   11010 -> [+1,-5]
    #   11011 -> [+5,-3]
    #   11100 -> [+3,+5]
    #   11101 -> [+5,+1]
    #   11110 -> [+1,+5]
    #   11111 -> [+5,+3]

    mapping_32qam = {
        (0,0,0,0,0): (-1,-1), (0,0,0,0,1): (-3,-1), (0,0,0,1,0): (-1,-3), (0,0,0,1,1): (-3,-3),
        (0,0,1,0,0): (-1,+1), (0,0,1,0,1): (-3,+1), (0,0,1,1,0): (-1,+3), (0,0,1,1,1): (-3,+3),
        (0,1,0,0,0): (+1,-1), (0,1,0,0,1): (+3,-1), (0,1,0,1,0): (+1,-3), (0,1,0,1,1): (+3,-3),
        (0,1,1,0,0): (+1,+1), (0,1,1,0,1): (+3,+1), (0,1,1,1,0): (+1,+3), (0,1,1,1,1): (+3,+3),
        (1,0,0,0,0): (-3,-5), (1,0,0,0,1): (-5,-1), (1,0,0,1,0): (-1,-5), (1,0,0,1,1): (-5,-3),
        (1,0,1,0,0): (-3,+5), (1,0,1,0,1): (-5,+1), (1,0,1,1,0): (-1,+5), (1,0,1,1,1): (-5,+3),
        (1,1,0,0,0): (+3,-5), (1,1,0,0,1): (+5,-1), (1,1,0,1,0): (+1,-5), (1,1,0,1,1): (+5,-3),
        (1,1,1,0,0): (+3,+5), (1,1,1,0,1): (+5,+1), (1,1,1,1,0): (+1,+5), (1,1,1,1,1): (+5,+3),
    }
    
    # Ensure the bitstream length is a multiple of 5
    if len(bitstream) % 5 != 0:
        bitstream = np.append(bitstream, np.zeros(5 - (len(bitstream) % 5)))

    # Reshape the bitstream into quintuplets of bits
    bitstream_quintuplets = bitstream.reshape(-1, 5)

    # Map the quintuplets to 32QAM symbols
    modulated_signal = np.zeros((bitstream_quintuplets.shape[0], 2))
    for _, quintuplet in enumerate(bitstream_quintuplets):
        modulated_signal[_, :] = mapping_32qam[tuple(quintuplet)]
        
    return modulated_signal

def modulate_64qam(bitstream: np.ndarray) -> np.ndarray:
    """
    Apply 64QAM modulation to the input signal.
    
    Parameters:
    - bitstream (np.ndarray): Input bitstream (1D array of bits).

    Returns:
    - modulated_signal (np.ndarray): Modulated signal (2D array with I and Q components).
    """
    # 64QAM mapping IQ:
    #   000000 -> [-7,-7]
    #   000001 -> [-7,-5]
    #   000010 -> [-7,-1]
    #   000011 -> [-7,-3]
    #   000100 -> [-7,+7]
    #   000101 -> [-7,+5]
    #   000110 -> [-7,+1]
    #   000111 -> [-7,+3]
    #   001000 -> [-5,-7]
    #   001001 -> [-5,-5]
    #   001010 -> [-5,-1]
    #   001011 -> [-5,-3]
    #   001100 -> [-5,+7]
    #   001101 -> [-5,+5]
    #   001110 -> [-5,+1]
    #   001111 -> [-5,+3]
    #   010000 -> [-1,-7]
    #   010001 -> [-1,-5]
    #   010010 -> [-1,-1]
    #   010011 -> [-1,-3]
    #   010100 -> [-1,+7]
    #   010101 -> [-1,+5]
    #   010110 -> [-1,+1]
    #   010111 -> [-1,+3]
    #   011000 -> [-3,-7]
    #   011001 -> [-3,-5]
    #   011010 -> [-3,-1]
    #   011011 -> [-3,-3]
    #   011100 -> [-3,+7]
    #   011101 -> [-3,+5]
    #   011110 -> [-3,+1]
    #   011111 -> [-3,+3]
    #   100000 -> [+7,-7]
    #   100001 -> [+7,-5]
    #   100010 -> [+7,-1]
    #   100011 -> [+7,-3]
    #   100100 -> [+7,+7]
    #   100101 -> [+7,+5]
    #   100110 -> [+7,+1]
    #   100111 -> [+7,+3]
    #   101000 -> [+5,-7]
    #   101001 -> [+5,-5]
    #   101010 -> [+5,-1]
    #   101011 -> [+5,-3]
    #   101100 -> [+5,+7]
    #   101101 -> [+5,+5]
    #   101110 -> [+5,+1]
    #   101111 -> [+5,+3]
    #   110000 -> [+1,-7]
    #   110001 -> [+1,-5]
    #   110010 -> [+1,-1]
    #   110011 -> [+1,-3]
    #   110100 -> [+1,+7]
    #   110101 -> [+1,+5]
    #   110110 -> [+1,+1]
    #   110111 -> [+1,+3]
    #   111000 -> [+3,-7]
    #   111001 -> [+3,-5]
    #   111010 -> [+3,-1]
    #   111011 -> [+3,-3]
    #   111100 -> [+3,+7]
    #   111101 -> [+3,+5]
    #   111110 -> [+3,+1]
    #   111111 -> [+3,+3]

    mapping_64qam = {
        (0,0,0,0,0,0): (-7,-7), (0,0,0,0,0,1): (-7,-5), (0,0,0,0,1,0): (-7,-1), (0,0,0,0,1,1): (-7,-3),
        (0,0,0,1,0,0): (-7,+7), (0,0,0,1,0,1): (-7,+5), (0,0,0,1,1,0): (-7,+1), (0,0,0,1,1,1): (-7,+3),
        (0,0,1,0,0,0): (-5,-7), (0,0,1,0,0,1): (-5,-5), (0,0,1,0,1,0): (-5,-1), (0,0,1,0,1,1): (-5,-3),
        (0,0,1,1,0,0): (-5,+7), (0,0,1,1,0,1): (-5,+5), (0,0,1,1,1,0): (-5,+1), (0,0,1,1,1,1): (-5,+3),
        (0,1,0,0,0,0): (-1,-7), (0,1,0,0,0,1): (-1,-5), (0,1,0,0,1,0): (-1,-1), (0,1,0,0,1,1): (-1,-3),
        (0,1,0,1,0,0): (-1,+7), (0,1,0,1,0,1): (-1,+5), (0,1,0,1,1,0): (-1,+1), (0,1,0,1,1,1): (-1,+3),
        (0,1,1,0,0,0): (-3,-7), (0,1,1,0,0,1): (-3,-5), (0,1,1,0,1,0): (-3,-1), (0,1,1,0,1,1): (-3,-3),
        (0,1,1,1,0,0): (-3,+7), (0,1,1,1,0,1): (-3,+5), (0,1,1,1,1,0): (-3,+1), (0,1,1,1,1,1): (-3,+3),
        (1,0,0,0,0,0): (+7,-7), (1,0,0,0,0,1): (+7,-5), (1,0,0,0,1,0): (+7,-1), (1,0,0,0,1,1): (+7,-3),
        (1,0,0,1,0,0): (+7,+7), (1,0,0,1,0,1): (+7,+5), (1,0,0,1,1,0): (+7,+1), (1,0,0,1,1,1): (+7,+3),
        (1,0,1,0,0,0): (+5,-7), (1,0,1,0,0,1): (+5,-5), (1,0,1,0,1,0): (+5,-1), (1,0,1,0,1,1): (+5,-3),
        (1,0,1,1,0,0): (+5,+7), (1,0,1,1,0,1): (+5,+5), (1,0,1,1,1,0): (+5,+1), (1,0,1,1,1,1): (+5,+3),
        (1,1,0,0,0,0): (+1,-7), (1,1,0,0,0,1): (+1,-5), (1,1,0,0,1,0): (+1,-1), (1,1,0,0,1,1): (+1,-3),
        (1,1,0,1,0,0): (+1,+7), (1,1,0,1,0,1): (+1,+5), (1,1,0,1,1,0): (+1,+1), (1,1,0,1,1,1): (+1,+3),
        (1,1,1,0,0,0): (+3,-7), (1,1,1,0,0,1): (+3,-5), (1,1,1,0,1,0): (+3,-1), (1,1,1,0,1,1): (+3,-3),
        (1,1,1,1,0,0): (+3,+7), (1,1,1,1,0,1): (+3,+5), (1,1,1,1,1,0): (+3,+1), (1,1,1,1,1,1): (+3,+3),
    }

    # Ensure the bitstream length is a multiple of 6
    if len(bitstream) % 6 != 0:
        bitstream = np.append(bitstream, np.zeros(6 - (len(bitstream) % 6)))

    # Reshape the bitstream into sextuplets of bits
    bitstream_sextuplets = bitstream.reshape(-1, 6)

    # Map the sextuplets to 64QAM symbols
    modulated_signal = np.zeros((bitstream_sextuplets.shape[0], 2))
    for _, sextuplet in enumerate(bitstream_sextuplets):
        modulated_signal[_, :] = mapping_64qam[tuple(sextuplet)]
    
    return modulated_signal