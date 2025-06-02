from keras.api.layers import *
from keras.src.models import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.regularizers import l2
from keras.api.optimizers import Adam


class LSTMClassifier:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
            LSTM(32, return_sequences = True, input_shape = input_shape, recurrent_dropout = 0.3),
            LSTM(16, recurrent_dropout = 0.3),
            Dense(32, activation = "relu", kernel_regularizer = l2(0.001)),
            Dense(num_classes, activation = "softmax")
        ])
        self.model.compile(
            optimizer = Adam(learning_rate = 0.001),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

    def train(self, x_train, y_train, batch_size = 32, epochs = 128):
        callbacks = [
            EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = True),
            ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 5, verbose = 1)
        ]
        history = self.model.fit(
            x_train, y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_split = 0.1,
            callbacks = callbacks,
            verbose = 1
        )
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose = 1)