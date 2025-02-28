import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загружаем более мощную модель GPT-2 (например, gpt2-large)
model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Функция для генерации ответа с улучшенными параметрами
def chat_with_gpt2(question):
    inputs = tokenizer.encode(question, return_tensors = "pt")
    
    # Генерация текста с более высоким качеством
    outputs = model.generate(inputs, 
                             max_length = 150, 
                             num_return_sequences = 1, 
                             no_repeat_ngram_size = 2, 
                             top_p = 0.95,          # Используем top-p для случайности
                             temperature = 0.7,     # Температура для контроля случайности
                             pad_token_id = tokenizer.eos_token_id,
                             early_stopping = True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return response

# Пример общения с улучшенной моделью
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == "выход":
        break
    response = chat_with_gpt2(user_input)
    print("Ответ: " + response)
