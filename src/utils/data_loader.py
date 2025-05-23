# Author: Anthony Yalong
# Description: This file contains the DataLoader class which is responsible for loading the data from the dataset and
# and preparing it for the model.

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Base dataset to be used
FILE_PATH = 'src/data/dataset.h5'

class DataLoader():
    """
    DataLoader class is responsible for loading the dataset and preparing it for the model.
    It loads the file from a h5py file and processes all the data to be represented in 
    a TensorFlow dataset. It also provides some utility functions to visualize the data.
    """
    # Main I/O charactersitics
    all_signals = []
    all_filter_taps = []

    # Shared characterstics
    all_bitstream_sizes = []
    all_modulation_types = []
    all_noise_levels = []
    all_filter_names = []
    all_signal_lengths = []

    # Known filter characteristics
    all_bitrates = []
    all_sampling_rates = []
    all_roll_offs = []

    # Random filter characteristics
    all_window_types = []

    # Dataset
    dataset = None

    def load_h5_dataset(self, file_path: str=FILE_PATH) -> None:
        """
        Load the dataset from the file path and store the data in the class `list` variables.

        Parameters
        ----------
        - file_path (str): The path to the dataset file.

        Returns
        -------
        - None
        """
        with h5py.File(file_path, 'r') as f:
            # Process Mmain I/O signals 
            self.all_signals = [sig.reshape(len(sig) // 2, 2) for sig in f['signals'][:]]
            self.all_filter_taps = f['filter_taps'][:]

            # Process shared characteristics
            self.all_bitstream_sizes = f['bitstream_sizes'][:]
            self.all_modulation_types = f['modulation_types'][:]
            self.all_noise_levels = f['noise_levels'][:]
            self.all_filter_names = f['filter_names'][:]
            self.all_signal_lengths = [len(sig) for sig in self.all_signals]

            # Process known filter characteristics
            self.all_bitrates = f['bitrates'][:]
            self.all_sampling_rates = f['sampling_rates'][:]
            self.all_roll_offs = f['roll_offs'][:]

            # Process random filter characteristics
            self.all_window_types = f['window_types'][:]


    def create_tf_dataset(self) -> None:
        """
        Create a TensorFlow dataset from the loaded data. The dataset is bucketed by sequence length
        to ensure that the model trains efficiently."

        Parameters
        ----------
        - None

        Returns
        -------
        - None
        """

        # Create a generator function to yield the data
        def generator():
            for sig, filter_taps, bitstream_size, modulation_type, noise_level, filter_name, signal_length, bitrate, sampling_rate, roll_off, window_type in zip(
                # Main I/O
                self.all_signals, 
                self.all_filter_taps, 

                # Shared characteristics
                self.all_bitstream_sizes, 
                self.all_modulation_types, 
                self.all_noise_levels, 
                self.all_filter_names,
                self.all_signal_lengths,

                # Known filter characteristics
                self.all_bitrates, 
                self.all_sampling_rates, 
                self.all_roll_offs, 

                # Random filter characteristics
                self.all_window_types
            ):
                yield sig, filter_taps, bitstream_size, modulation_type, noise_level, filter_name, signal_length, bitrate, sampling_rate, roll_off, window_type
        
        # Create the dataset from the generator
        self.dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        ).prefetch(tf.data.AUTOTUNE)

        # Bucket the dataset by signal length
        self.dataset = self.dataset.bucket_by_sequence_length(
            element_length_func=lambda *args: tf.shape(args[0])[0],
            bucket_boundaries=[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
            bucket_batch_sizes=[64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2],
            padded_shapes=(
                tf.TensorShape([None, 2]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([])
            )
        )

    def plot_raw(self, num_samples: int=10) -> None:
        """
        Plot the raw signals, I and Q components, and FIR filter taps for a given number of samples.

        Parameters
        ----------
        - num_samples (int): Number of samples to plot.

        Returns
        -------
        - None
        """
        for _ in range(num_samples):            
            # Plot I and Q components and FIR filter taps
            plt.figure(figsize=(10, 6))

            plt.subplot(3, 1, 1)
            plt.plot(self.all_signals[_][:, 0], label='I Component')
            plt.title('I Component')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(self.all_signals[_][:, 1], label='Q Component', color='orange')
            plt.title('Q Component')
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.stem(self.all_filter_taps[_], label='FIR Filter Taps')
            plt.title('FIR Filter Taps')
            plt.grid(True)
            plt.suptitle(f'Modulation: {self.all_modulation_types[_]}, Filter: {self.all_filter_names[_]}, Noise Level: {self.all_noise_levels[_]:.2f}, Bitstream Size: {self.all_bitstream_sizes[_]}, Num Taps: {len(self.all_filter_taps[_])}')
            plt.tight_layout()
            plt.show()
            plt.close()
    
    def plot_batch(self, dataset=None, samples_per_batch: int=4) -> None:
        """
        Plot `samples_per_batch` number of samples from each batch in the dataset.

        Parameters
        ----------
        - samples_per_batch (int): Number of samples to plot per batch.

        Returns
        -------
        - None
        """
        if dataset is None:
            dataset = self.dataset

        # Loop through batches
        for batch in dataset.take(samples_per_batch):
            signals, fir_filters, _, modulation_types, noise_levels, _, signal_length, _, _, _, _ = batch

            # Loop through samples in batch
            for sample in range(len(signals)):
                # Preprocess for plotting
                signal = signals[sample].numpy()
                fir_filter = fir_filters[sample].numpy()

                plt.figure(figsize=(10, 6))

                plt.subplot(3, 1, 1)
                plt.plot(signal[:, 0], label="Real Part")
                plt.title("Signal for First Sample")

                plt.subplot(3, 1, 2)
                plt.plot(signal[:, 1], label="Imaginary Part")
                plt.title("Signal for First Sample")

                plt.subplot(3, 1, 3)
                plt.plot(fir_filter, label="FIR Filter Taps")
                plt.title("FIR Filter Taps for First Sample")

                plt.suptitle(f"Modulation: {modulation_types[sample].numpy().decode()}, Noise Level: {noise_levels[sample].numpy():.2f}, Signal Length: {signal_length[sample].numpy()}, Bucket Length: {len(signal)}")
                plt.tight_layout()
                plt.show()

    def plot_prediction(self, input: np.ndarray, predictions: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Plot the predictions and ground truth for a given batch of data.

        Parameters
        ----------
        - input (np.ndarray): The input data for the model.
        - predictions (np.ndarray): The predicted values.
        - ground_truth (np.ndarray): The ground truth values.

        Returns
        -------
        - None
        """
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(input[:, 0], label='Input I Component', color='green')
        plt.plot(input[:, 1], label='Input Q Component', color='red')
        plt.title('Input Signal')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(predictions, label='Predictions', color='orange')
        plt.plot(ground_truth, label='Ground Truth', color='blue')
        plt.title('Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True)
        plt.show()
                
    def get_dataset(self) -> tf.data.Dataset:
        """
        Return the dataset.

        Parameters
        ----------
        - None

        Returns
        -------
        - tf.data.Dataset: The dataset.
        """
        return self.dataset
    
    def reset(self) -> None:
        """
        Reset the DataLoader.

        Parameters
        ----------
        - None

        Returns
        -------
        - None
        """
        self.all_signals = []
        self.all_filter_taps = []

        self.all_bitstream_sizes = []
        self.all_modulation_types = []
        self.all_noise_levels = []
        self.all_filter_names = []
        self.all_signal_lengths = []

        self.all_bitrates = []
        self.all_sampling_rates = []
        self.all_roll_offs = []

        self.all_window_types = []

        self.dataset = None

if __name__ == "__main__":
    dataloader = DataLoader()

    dataloader.load_h5_dataset("src/data/dataset_3c.h5")
    dataloader.create_tf_dataset()
    dataset = dataloader.get_dataset()
    dataloader.plot_batch()