import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.api.utils import to_categorical
from sklearn.model_selection import train_test_split
import Time_Step


def prepare_data(
    filepath,                     # path to the CSV file (e.g., 'PID_1_BPM_cleaned.csv')
    num_classes = 17,               # number of classes for one-hot encoding (e.g., 17 for BPM from 0 to 16)
    time_step = 5,                  # length of the time window for LSTM
    train_ratio = 0.8,              # ratio of data used for training (e.g., 0.8 for 80% train, 20% test)
    sampling_step = 1
):
    
    # Read the CSV file
    df = pd.read_csv(filepath)

    # --- VERY IMPORTANT: CHOOSE THE CORRECT DATA LOADING OPTION ---
    # Option 1: If the data in the CSV is SORTED FROM OLDEST TO NEWEST records
    # (normal chronological order), use this line:
    data = df.to_numpy()

    # Option 2: If the data in the CSV is SORTED FROM NEWEST TO OLDEST records
    # (reverse chronological order), use this line:
    # data = np.flip(df.to_numpy(), axis=0)

    # Convert all values to float32
    data = data.astype(np.float32)

    # Subsample the data using the given step
    data = data[:: sampling_step]

    # Scale the entire dataframe to the [0, 1] range
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Split the scaled data into features (x) and labels (y)
    # It is assumed that the target label (BPM) is in the last column
    x = data_scaled[:, :-1]  # All columns except the last one are features
    y = data_scaled[:, -1]   # The last column is the label (BPM)

    # Create time windows (sequences) for features and labels
    x_seq, y_seq = Time_Step.create_dataset(x, y, time_step)

    # Convert the scaled labels (y_seq) into integer classes (0 to num_classes-1)
    # and then to one-hot encoded format.
    # Important: y_seq is first multiplied by (num_classes - 1) to get values from 0 to 16,
    # then cast to int, and only then one-hot encoded.
    y_seq = (y_seq * (num_classes - 1)).astype(int)
    y_seq = to_categorical(y_seq, num_classes = num_classes)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_seq, y_seq, test_size = 1 - train_ratio, random_state = 42
    )

    return x_train, y_train, x_test, y_test
