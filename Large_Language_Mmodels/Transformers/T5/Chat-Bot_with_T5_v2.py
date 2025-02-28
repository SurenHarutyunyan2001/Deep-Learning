import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Загружаем модель T5-large
model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def chat_with_t5(question):
    inputs = tokenizer.encode("question: " + question, return_tensors="pt")
    
    # Генерация с улучшенными параметрами
    outputs = model.generate(inputs, 
                             max_length = 150, 
                             num_return_sequences = 1,
                             top_p = 0.95,           # top-p для случайности
                             temperature = 0.7,      # Температура для регулирования случайности
                             no_repeat_ngram_size = 2,
                             early_stopping = True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return response

# Пример общения с улучшенной моделью T5
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == "выход":
        break
    response = chat_with_t5(user_input)
    print("Ответ: " + response)
