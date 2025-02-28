import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Загружаем модель и токенизатор GPT-2 для TensorFlow
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Устанавливаем pad_token_id, если его нет
model.config.pad_token_id = model.config.eos_token_id

# Передаем пустой ввод (например, пробел)
input_text = " "
input_ids = tokenizer.encode(input_text, return_tensors="tf")

# Генерация текста с улучшенными параметрами
output = model.generate(
    input_ids,
    max_length = 100,              # Максимальная длина текста
    num_return_sequences = 1,      # Количество вариантов текста
    no_repeat_ngram_size = 2,      # Избегать повторений
    temperature = 0.7,             # Контролирует случайность текста
    top_p = 0.9,                   # Ограничение по вероятности
    top_k = 50                     # Ограничение на количество токенов
)

# Декодируем результат обратно в текст
generated_text = tokenizer.decode(output[0], skip_special_tokens = True)

print("Generated Text: ", generated_text)
