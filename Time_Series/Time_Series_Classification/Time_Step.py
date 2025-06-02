import tensorflow as tf
import numpy as np

def make_n_time_step_tf(x, y, time_steps = 5, to_numpy = True):
    """
    Creates sequences of time windows from input data using TensorFlow.

    Parameters:
    - time_steps: length of the time window (number of time steps)
    - x: input feature array (numpy array), shape (samples, features) or (samples,)
    - y: label array, similar in length to x
    - to_numpy: if True, returns numpy arrays; otherwise, returns tf.Tensors

    Returns:
    - x: array of time windows, shape (num_windows, time_steps, features)
    - y: array of labels corresponding to each window (the next value after the window)
    """
    
    # Convert input data to TensorFlow tensors with float32 dtype
    x = tf.convert_to_tensor(x, dtype = tf.float32)
    y = tf.convert_to_tensor(y, dtype = tf.float32)

    # frame_length = window length, frame_step = step between windows (default is 1)
    x = tf.signal.frame(x, frame_length = time_steps, frame_step = 1, axis = 0)

    # Each sequence corresponds to the next label in time from y
    y = y[time_steps :]

    # Return either tensors or numpy arrays depending on the flag
    if to_numpy:
        return x.numpy(), y.numpy()
    return x, y


def create_dataset(x, y, time_steps = 5):
    """
    Creates time windows from data x and corresponding labels from y.

    Parameters:
    - x: feature array (numpy array), shape (samples, features)
    - y: label array (numpy array), shape (samples,)
    - time_steps: length of the time window

    Returns:
    - xs: array of input sequences (samples, time_steps, features)
    - ys: array of labels (samples,)
    """
    xs, ys = []

    # Forming windows and corresponding labels
    for i in range(len(x) - time_steps):
        xs.append(x[i:(i + time_steps)])     # time window of features
        ys.append(y[i + time_steps])         # label — the next value after the window

    return np.array(xs), np.array(ys)
