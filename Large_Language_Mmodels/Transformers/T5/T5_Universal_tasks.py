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
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Примеры использования
if __name__ == "__main__":
    tasks = [
        # Перевод текста
        ("translate English to French: How are you?", "Translation"),
        
        # Суммаризация текста
        ("summarize: This is a long paragraph describing many details about a certain topic. The summary should capture the main idea concisely.", "Summarization"),
        
        # Вопрос-ответ
        ("question: What is the capital of France? context: France is a country in Europe. Its capital is Paris.", "Question Answering"),
        
        # Классификация текста
        ("classify sentiment: I love this product, it is amazing!", "Sentiment Classification"),
        
        # Распознавание именованных сущностей
        ("extract entities: Barack Obama was born in Hawaii.", "Named Entity Recognition"),
        
        # Объяснение текста
        ("explain: Why does the sun rise in the east?", "Explanation"),
        
        # Генерация кода
        ("write Python function: a function to calculate factorial", "Code Generation"),
        
        # Заполнение пропусков в тексте
        ("fill in the blanks: The capital of France is <extra_id_0>.", "Fill in the Blanks"),
        
        # Переписывание текста в другом стиле
        ("convert to formal: Hey, what's up?", "Style Conversion")
    ]

    for task, task_name in tasks:
        print(f"\nTask: {task_name}")
        result = generate_text(task)
        print("Input:", task)
        print("Output:", result)


