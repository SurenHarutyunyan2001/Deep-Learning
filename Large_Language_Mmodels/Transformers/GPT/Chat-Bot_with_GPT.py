import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загружаем модель и токенизатор GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Функция для генерации ответа на основе введённого вопроса
def chat_with_gpt2(question):
    # Преобразуем вопрос в токены
    inputs = tokenizer.encode(question, return_tensors = "pt")
    
    # Генерация текста (ответа)
    outputs = model.generate(inputs, max_length = 100,
                             num_return_sequences = 1,
                             no_repeat_ngram_size = 2,
                             pad_token_id=tokenizer.eos_token_id)
    
    # Преобразуем токены обратно в текст
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    
    return response

# Пример общения с моделью
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == "выход":
        break
    response = chat_with_gpt2(user_input)
    print("Ответ: " + response)
