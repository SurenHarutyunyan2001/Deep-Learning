import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import sys
import sqlite3
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Создание или подключение к базе данных SQLite
conn = sqlite3.connect('documents.db')
cursor = conn.cursor()

# Создание таблицы для документов, если она не существует
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    text TEXT
)
''')

# Заполнение базы данных документами (если таблица пуста)
documents = [
    "Эверест — самая высокая гора в мире, высота которой составляет 8848 метров.",
    "Гора Эверест расположена в Гималаях на границе Непала и Тибета.",
    "Вулкан Фудзи — это символ Японии, его высота составляет 3776 метров.",
    "Эверест также известен как Джомолунгма, его высота 8848,86 метров.",
    "Гора Эверест является одной из самых популярных гор для альпинистов, однако восхождение на неё является очень опасным.",
    "Многие люди погибают при попытке подняться на вершину Эвереста из-за экстремальных условий и недостатка кислорода."
]

# Вставка документов в таблицу, если таблица пуста
cursor.execute("SELECT COUNT(*) FROM documents")
count = cursor.fetchone()[0]
if count == 0:
    cursor.executemany("INSERT INTO documents (text) VALUES (:text)", [{"text": doc} for doc in documents])
    conn.commit()

# Загрузка модели для эмбеддингов
model = SentenceTransformer('all-MiniLM-L12-v2')  # Модель для создания эмбеддингов

# Загрузка модели GPT-2 для генерации ответов
tokenizer_gen = GPT2Tokenizer.from_pretrained("gpt2")
model_gen = GPT2LMHeadModel.from_pretrained("gpt2")

# Установка токена для паддинга
tokenizer_gen.pad_token = tokenizer_gen.eos_token

# Генерация эмбеддингов для документов
cursor.execute("SELECT text FROM documents")
documents_from_db = [row[0] for row in cursor.fetchall()]
doc_embeddings = model.encode(documents_from_db, convert_to_tensor=True)

# Функция поиска наиболее релевантных документов с использованием FAISS
def retrieve(query, k=3):
    query_embedding = model.encode(query, convert_to_tensor = True)
    scores = util.cos_sim(query_embedding, doc_embeddings)
    
    # Преобразуем результат в массив и сортируем
    scores = scores.cpu().detach().numpy()
    top_k_indices = scores.argsort()[0][-k:][::-1]
    
    # Извлекаем документы по индексам
    retrieved_docs = [documents_from_db[i] for i in top_k_indices]
    return retrieved_docs

# Генерация ответа с использованием GPT-2
def generate_answer(query, retrieved_docs):
    input_text = f"Вопрос: {query} Контекст: {' '.join(retrieved_docs)}"
    
    # Преобразуем текст в формат GPT-2
    inputs = tokenizer_gen(input_text, return_tensors="pt", padding = True, truncation = True)
    
    # Генерация ответа с использованием GPT-2
    outputs = model_gen.generate(
        inputs["input_ids"],
        max_new_tokens = 150,  # Максимальное количество новых токенов
        num_beams = 5,  # Использование beam search для более качественного ответа
        temperature = 0.7,  # Регулировка случайности
        top_p = 0.9,  # Настройка вероятностного сэмплинга
        no_repeat_ngram_size = 2,  # Удаление повторяющихся фраз
        pad_token_id=tokenizer_gen.eos_token_id  # Установить pad_token_id
    )
    
    # Декодируем и возвращаем результат
    return tokenizer_gen.decode(outputs[0], skip_special_tokens = True)

# Пример
query = "Какая высота Эвереста?"
retrieved_docs = retrieve(query)
answer = generate_answer(query, retrieved_docs)

print("Запрос:", query)
print("Ответ:", answer)

# Закрытие соединения
conn.close()
