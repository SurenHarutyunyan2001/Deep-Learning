import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf

# Загрузка токенизатора и модели
model_name = "t5-small"  # Можно заменить на "t5-base" или другую версию

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

# Функция для генерации текста
def generate_text(input_text, max_length=50):
    """
    Генерирует текст на основе входной строки с использованием T5.

    :param input_text: Исходный текст
    :param max_length: Максимальная длина сгенерированного текста
    :return: Сгенерированный текст
    """
    # Препроцессинг входного текста
    input_ids = tokenizer.encode(input_text, return_tensors="tf")

    # Генерация текста
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,  # Использование beam search
        early_stopping=True
    )

    # Декодирование результата
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return generated_text

# Пример использования
if __name__ == "__main__":
    input_prompt = "translate English to French: How are you?"
    result = generate_text(input_prompt)
    print("\nInput:", input_prompt)
    print("Generated Text:", result)
