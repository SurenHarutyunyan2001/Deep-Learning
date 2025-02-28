import keras
from keras import layers
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import RandomSearch

# Входные данные
(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.
# Определение гиперпараметров
hp = HyperParameters()
hp.Choice('learning_rate', [0.1, 0.001])
hp.Int('num_layers', 2, 10)

# Определение модели с изменяемым количеством слоев
def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Flatten(input_shape = (28, 28)))
    for _ in range(hp.get('num_layers')):
        model.add(layers.Dense(32, activation = 'relu'))

    model.add(layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer = keras.optimizers.Adam(hp.get('learning_rate')), 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    return model

hypermodel = RandomSearch(
    build_model,
    max_trials = 20, # Количество допустимых комбинаций
    hyperparameters = hp,
    allow_new_entries = False,
    objective = 'val_accuracy')

hypermodel.search(
    x = x,
    y = y,
    epochs = 5,
    validation_data = (val_x, val_y))

# Показать параметры, соответствующие лучшей модели
hypermodel.results_summary()