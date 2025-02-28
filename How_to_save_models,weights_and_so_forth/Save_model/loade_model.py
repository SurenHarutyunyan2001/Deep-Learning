import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from tensorflow.keras.models import load_model
import numpy as np

# Загрузка сохраненной модели
loaded_model = load_model('simple_model.keras')
print("Model loaded successfully.")

# Использование загруженной модели для предсказаний
predictions = loaded_model.predict(np.array([[6], [7], [8]]))

# Вывод предсказаний
print("Predictions:", predictions.flatten())
