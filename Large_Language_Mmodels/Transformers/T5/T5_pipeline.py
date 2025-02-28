from transformers import pipeline

# Настройка пайплайна для генерации с объяснением
model_name = "google/flan-t5-xl"  # Модель, подходящая для CoT
model = pipeline("text-generation", model = model_name)

input_text = "Если у Маши 3 яблока и она купила еще 2, сколько у нее яблок?"
result = model(input_text, max_length = 100, num_return_sequences = 1)

print(result[0]['generated_text'])
