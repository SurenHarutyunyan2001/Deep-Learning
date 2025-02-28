import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Загружаем модель T5 и её токенизатор
model_name = "t5-small"  # Используем компактную модель для демонстрации
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Пример входных данных для двух задач
tasks = [
    {"task": "translate English to French", "input_text": "The weather is nice today."},
    {"task": "summarize", "input_text": "Machine learning is a subfield of artificial intelligence that involves training algorithms on data to make predictions or decisions."}
]

# Функция для выполнения задач с использованием T5
def perform_task(task, input_text):
    # Формируем запрос с префиксом задачи
    input_with_prefix = f"{task}: {input_text}"
    inputs = tokenizer(input_with_prefix, return_tensors = "pt", padding = True)
    
    # Генерация текста
    outputs = model.generate(
        inputs["input_ids"],
        max_length = 50,  # Ограничиваем длину ответа
        num_beams = 4,    # Используем beam search для улучшения качества
        early_stopping = True
    )
    
    # Декодируем результат
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

# Выполняем задачи
for task_data in tasks:
    task = task_data["task"]
    input_text = task_data["input_text"]
    result = perform_task(task, input_text)
    print(f"Task: {task}\nInput: {input_text}\nOutput: {result}\n")

