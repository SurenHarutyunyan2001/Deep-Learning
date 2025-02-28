import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Загружаем модель и токенизатор T5
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def chat_with_t5(question):
    # Преобразуем вопрос в токены
    inputs = tokenizer.encode("question: " + question, return_tensors = "pt")
    
    # Генерация текста (ответа)
    outputs = model.generate(inputs, max_length = 100, num_return_sequences = 1)
    
    # Преобразуем токены обратно в текст
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    
    return response

# Пример общения с моделью
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == "выход":
        break
    response = chat_with_t5(user_input)
    print("Ответ: " + response)