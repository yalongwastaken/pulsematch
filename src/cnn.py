"""
Author: Anthony Yalong
Description: This script sets up the PulseMatch 1D-Convolutional Neural Network (CNN) model and is designed to be executed as a 
standalone Python script. The setup includes all necessary components for initializing, training, and evaluating the model. For 
detailed results and experimental analysis, please refer to the notebooks directory where each respective model was actually 
trained.
"""

# Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader


# Load the Dataset
FILE_PATH_A = "/content/drive/MyDrive/academics/gwu/2024-2025/spring/CSCI 4366/data/dataset_1a.h5"
FILE_PATH_B = "/content/drive/MyDrive/academics/gwu/2024-2025/spring/CSCI 4366/data/dataset_1b.h5"
FILE_PATH_C = "/content/drive/MyDrive/academics/gwu/2024-2025/spring/CSCI 4366/data/dataset_1c.h5"

dataloader = DataLoader()

# Full dataset
dataset = None

# Load file A
dataloader.load_h5_dataset(FILE_PATH_A)
dataloader.create_tf_dataset()
dataset = dataloader.get_dataset()

# Load file B
dataloader.load_h5_dataset(FILE_PATH_B)
dataloader.create_tf_dataset()
dataset = dataset.concatenate(dataloader.get_dataset())

# Load file C
dataloader.load_h5_dataset(FILE_PATH_C)
dataloader.create_tf_dataset()
dataset = dataset.concatenate(dataloader.get_dataset())

# Split dataset 8:1:1
dataset = dataset.shuffle(50000)

dataset_len = sum(1 for _ in dataset)
train_size = int(0.8 * dataset_len)
val_size = int(0.1 * dataset_len)
test_size = dataset_len - train_size - val_size

print(f"Dataset length: {dataset_len}")
print(f"Training size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

dataset_train = dataset.take(train_size)
dataset_val = dataset.skip(train_size).take(val_size)
dataset_test = dataset.skip(train_size + val_size).take(test_size)

# Visualize Data
dataloader.plot_batch(dataset_train)

# Preprocessing function
def preprocess_data(dataset):
  def map_fn(signals, fir_filters, *_):
    return signals, fir_filters

  return dataset.map(map_fn)

# Apply preprocessing to the datasets
dataset_train = preprocess_data(dataset_train)
dataset_val = preprocess_data(dataset_val)
dataset_test = preprocess_data(dataset_test)

# Examine
print(next(iter(dataset_train)))

# PulseMatch 1D-CNN Class
class PulseMatchCNN(tf.keras.Model):
  def __init__(
    self,
    num_dense_layers: int = 3,
    dense_layer_size: int = 64,
    dropout: float = 0.1,
    non_zero_weight: float = 100.0,
    kernel_regularizer: tf.keras.regularizers.Regularizer = None
  ) -> None:
    super().__init__()

    # Model parameters
    self.num_dense_layers = num_dense_layers
    self.dense_layer_size = dense_layer_size
    self.dropout = dropout
    self.non_zero_weight = non_zero_weight
    self.kernel_regularizer = kernel_regularizer

    # Build the model
    self.model = self.make_model()

  def make_model(self) -> tf.keras.Model:
    """
    Create PulseMatchCNN model with convolutional layers followed by dense layers.
    """
    inputs = tf.keras.Input(shape=(None, 2))

    # Convolutional layers
    x = self._conv_block(inputs, 8, 128)
    x = self._conv_block(x, 32, 64)
    x = self._conv_block(x, 128, 32)
    x = self._conv_block(x, 256, 16)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    for _ in range(self.num_dense_layers):
      x = tf.keras.layers.Dense(
        self.dense_layer_size,
        activation='relu',
        kernel_regularizer=self.kernel_regularizer
      )(x)
      x = tf.keras.layers.Dropout(self.dropout)(x)

    # Output heads
    fir_output = tf.keras.layers.Dense(432, activation='tanh', name='fir_output')(x)

    return tf.keras.Model(inputs=inputs, outputs=fir_output)

  def _conv_block(self, x, filters: int, kernel_size: int) -> tf.Tensor:
    """
    Helper function to create a convolution block with BatchNormalization,
    ReLU activation, and MaxPooling1D.
    """
    x = tf.keras.layers.Conv1D(filters, kernel_size, activation=None, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    return x

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    return self.model(inputs=inputs, training=training)

  def compute_loss(
    self,
    x: tf.Tensor,
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    sample_weight: tf.Tensor = None
  ) -> tf.Tensor:
    """
    Compute loss function with higher weighting for non-zero values.
    """
    # Compute MSE
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    weight_mask = 1.0 + (mask * (self.non_zero_weight - 1.0))
    fir_loss = tf.reduce_sum(weight_mask * tf.square(y_true - y_pred)) / (tf.reduce_sum(weight_mask) + 1e-8)

    # Combined losses
    total_loss = fir_loss

    if sample_weight is not None:
      total_loss = tf.reduce_sum(total_loss * sample_weight) / (tf.reduce_sum(sample_weight) + 1e-8)

    return total_loss

  def save_model(self, path: str) -> None:
    self.model.save(path)

  def load_model(self, path: str) -> None:
      self.model = tf.keras.models.load_model(path)
      print(f"Full model loaded from {path}")

# Initialize model
pulsematchCNN = PulseMatchCNN(
  num_dense_layers=3,
  dense_layer_size=64,
  dropout=0.25,
  non_zero_weight=100.0,
  kernel_regularizer=tf.keras.regularizers.l2(0.0001)
)

# Decaying Learning Rate
initial_lr = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate=initial_lr,
  decay_steps=1000,
  decay_rate=0.96,
  staircase=True
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
pulsematchCNN.compile(optimizer=optimizer, loss=pulsematchCNN.compute_loss)

# Model callbacks
callbacks = [
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
  tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# Train the model
history = pulsematchCNN.fit(
  dataset_train,
  epochs=100,
  steps_per_epoch=256,
  callbacks=callbacks,
  validation_data=dataset_val,
  validation_steps=128
)

# Visualize history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print loss arrays
print(f"Loss: {loss}")
print(f"Validation Loss: {val_loss}")

# Plot loss arrays
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()

# Set batch index for fetching the batch
batch_index = 1
dataset_batch = dataset_test.skip(batch_index).take(1)

dataset_test_x, dataset_test_y = next(iter(dataset_batch))

# Get predictions
predictions = pulsematchCNN.predict(dataset_test_x)

# Plot predictoins
for i in range(len(predictions)):
  plt.figure(figsize=(10, 6))

  # First subplot: Predicted FIR filter values
  plt.subplot(2, 1, 1)
  plt.plot(predictions[i], label='Predictions', color='orange')
  plt.title(f'FIR Filter Tap Prediction for Batch: {batch_index}, Sample: {i}')
  plt.ylim(-1, 1)
  plt.grid(True)

  # Second subplot: Ground truth FIR filter values
  plt.subplot(2, 1, 2)
  plt.plot(dataset_test_y[i], label='Ground Truth', color='blue')
  plt.title(f'Ground Truth Batch: {batch_index}, Sample: {i}')
  plt.ylim(-1, 1)
  plt.grid(True)

  plt.tight_layout()
  plt.show()

# Weighted MSE Loss
weighted_mse_loss = pulsematchCNN.compute_loss(dataset_test_x, dataset_test_y, predictions)
print(f"Weighted MSE Loss: {weighted_mse_loss}")

# Raw MSE Loss
raw_mse_loss = tf.reduce_mean(tf.square(dataset_test_y - predictions))
print(f"Raw MSE Loss: {raw_mse_loss}")

# Thresholds
thresholds = np.linspace(0.10, 0, num=21)

# Prepare arrays to store accuracies
original_accuracies = []
weighted_accuracies = []
nonzero_accuracies = []

# Errors and weights
errors = tf.abs(dataset_test_y - predictions)
weights = tf.abs(dataset_test_y)

# Original accuracy (unweighted, on all taps)
for threshold in thresholds:
  correct = tf.cast(errors < threshold, tf.float32)
  acc = tf.reduce_mean(correct) * 100
  original_accuracies.append(acc.numpy())

# Weighted accuracy
for threshold in thresholds:
  correct = tf.cast(errors < threshold, tf.float32)
  weighted_correct = correct * weights
  weighted_total = weights + 1e-8  # to prevent divide-by-zero
  acc = tf.reduce_sum(weighted_correct) / tf.reduce_sum(weighted_total) * 100
  weighted_accuracies.append(acc.numpy())

# Non-zero tap accuracy
non_zero_mask = tf.not_equal(dataset_test_y, 0.0)
filtered_errors = tf.boolean_mask(errors, non_zero_mask)

for threshold in thresholds:
  correct = tf.cast(filtered_errors < threshold, tf.float32)
  acc = tf.reduce_mean(correct) * 100
  nonzero_accuracies.append(acc.numpy())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(thresholds, original_accuracies, marker='o', label='Original Accuracy')
plt.plot(thresholds, weighted_accuracies, marker='s', label='Weighted Accuracy')
plt.plot(thresholds, nonzero_accuracies, marker='^', label='Non-zero Tap Accuracy')

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Threshold Accuracy Percentage (%)', fontsize=12)
plt.title('Threshold Accuracy Percentage vs. Threshold', fontsize=14)
plt.gca().invert_xaxis()
plt.xticks(np.round(np.arange(0.10, -0.001, -0.01), 3), fontsize=10, rotation=-45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()