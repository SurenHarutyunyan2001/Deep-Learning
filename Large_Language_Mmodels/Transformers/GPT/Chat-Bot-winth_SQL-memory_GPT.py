import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import sqlite3
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Инициализация базы данных SQLite
conn = sqlite3.connect('chat_memory.db')
cursor = conn.cursor()

# Создание таблицы для хранения истории общения (если она не существует)
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    user_input TEXT,
                    bot_response TEXT)''')
conn.commit()

# Загружаем модель GPT-2
model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Функция для записи общения в базу данных
def save_chat_history(user_input, bot_response):
    cursor.execute("INSERT INTO chat_history (user_input, bot_response) VALUES (?, ?)",
                   (user_input, bot_response))
    conn.commit()

# Функция для извлечения истории общения из базы данных
def get_chat_history():
    cursor.execute("SELECT user_input, bot_response FROM chat_history")
    rows = cursor.fetchall()
    return [f"User: {row[0]}\nBot: {row[1]}" for row in rows]

# Функция для генерации ответа с учётом долговременной памяти
def chat_with_gpt2(question):
    # Получаем всю историю общения
    history = "\n".join(get_chat_history())
    
    # Добавляем новый вопрос в историю
    full_input = history + f"\nUser: {question}\nBot:"
    
    # Преобразуем текст в токены
    inputs = tokenizer.encode(full_input, return_tensors="pt")
    
    # Генерация текста с более высоким качеством
    outputs = model.generate(inputs, 
                             max_length = 150, 
                             num_return_sequences = 1, 
                             no_repeat_ngram_size = 2, 
                             top_p = 0.95,         
                             temperature = 0.7,    
                             pad_token_id = tokenizer.eos_token_id,
                             early_stopping = True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Сохраняем запрос и ответ в базе данных
    save_chat_history(question, response)
    
    return response

# Пример общения с улучшенной моделью и долговременной памятью
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == "выход":
        break
    response = chat_with_gpt2(user_input)
    print("Ответ: " + response)

# Закрытие соединения с базой данных
conn.close()
