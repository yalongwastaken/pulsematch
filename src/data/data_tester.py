import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with h5py.File('test_dataset.h5', 'r') as f:
        # Check the keys in the HDF5 file
        print("Keys in the HDF5 file:", list(f.keys()))

        # Read the I and Q signals from the HDF5 file
        signals = f['signals'][:]
        signals_i = f['signals_i'][:] # Shape (M, N)
        signals_q = f['signals_q'][:] # Shape (M, N)

        print("Signals shape:", signals.shape)
        print("I signals shape:", signals_i.shape)
        print("Q signals shape:", signals_q.shape)

        # WAY ONE
        all_signals_1 = [sig.reshape(2, len(sig) // 2) for sig in signals]

        # Check the shape of the signals
        for i in range(len(signals)):
            print(f"I signal{i} shape:", signals_i[i].shape)
            print(f"Q signals{i} shape:", signals_q[i].shape)
            print("Signals shape after reshape:", all_signals_1[i].shape)


        # WAY TWO
        all_signals_2 = []

        for _ in range(len(signals_i)):
            # Stack the I and Q signals
            signals = np.stack((signals_i[_], signals_q[_]), axis=-1)
            all_signals_2.append(signals)


        # Check first values for equality
        for i in range(len(all_signals_1)):
            print(f"Signal {i}:")
            print("WAY ONE:", all_signals_1[i][:3])
            print("WAY TWO:", all_signals_2[i][:3])
            print("Equal:", np.array_equal(all_signals_1[i], all_signals_2[i]))

        # Check sizes of each dataset
        print(f"Size of signals: {len(all_signals_2)}")
        
        for _ in f:
            print(f"Size of Dataset Feature {_}: {f[_].shape}")


        for i in range(len(all_signals_2)):
            # plot the first signal
            plt.figure(figsize=(10, 5))
            plt.plot(all_signals_2[i][:, 0], label='I signal')
            plt.plot(all_signals_2[i][:, 1], label='Q signal')
            plt.title('I and Q signals')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()
            plt.show()

        for i in range(len(signals)):
            # plot the first signal
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(all_signals_1[i][:, 0], label='I signal: WAY ONE')
            plt.plot(all_signals_2[i][:, 1], label='I signal: WAY TWO')
            plt.title('I Comparison')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.plot(all_signals_1[i][:, 1], label='Q signal: WAY ONE')
            plt.plot(all_signals_2[i][:, 0], label='Q signal: WAY TWO')
            plt.title('Q Comparison')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')

            plt.legend()
            plt.grid()
            plt.suptitle(f'Signal {i+1} Comparison')
            plt.tight_layout()
            plt.show()

