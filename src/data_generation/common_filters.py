# Author: Anthony Yalong
# Description: Script file to generate common filters (RRC, RC, SINC, GAUSSIAN) for PulseMatch.

import numpy as np

# Roll-off factors
roll_off_factors = [
    0.125,
    0.250,
    0.500,
    0.750,
    1.000,
]

def rrc(num_taps: int, symbol_rate: int) -> np.ndarray:
    """Generate a root raised cosine (RRC) filter."""
    # Ensure num_taps is odd
    if num_taps % 2 == 0:
        num_taps += 1

    # Time index array
    time_indices = np.arange(-num_taps // 2, num_taps // 2 + 1)
    
    # Roll-off factor (ß)
    roll_off_factor = np.random.choice(roll_off_factors)
    
    # Symbol period
    symbol_period = 1
    
    # Standard RRC formula
    rrc_filter = np.sinc(time_indices / symbol_period) * \
                 np.cos(np.pi * roll_off_factor * time_indices / symbol_period) / \
                 (1 - (4 * roll_off_factor * time_indices / symbol_period) ** 2)

    # Normalize the filter to ensure the sum of its values equals 1
    normalized_rrc_filter = rrc_filter / np.sum(rrc_filter)
    
    return normalized_rrc_filter