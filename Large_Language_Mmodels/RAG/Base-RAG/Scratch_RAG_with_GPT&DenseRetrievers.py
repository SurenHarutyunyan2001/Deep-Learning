import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загрузка модели для генерации эмбеддингов
model = SentenceTransformer('all-MiniLM-L12-v2')

# Загрузка модели генерации GPT-2
tokenizer_gen = GPT2Tokenizer.from_pretrained("gpt2")
model_gen = GPT2LMHeadModel.from_pretrained("gpt2")

# Устанавливаем pad_token равным eos_token для токенизатора генерации
tokenizer_gen.pad_token = tokenizer_gen.eos_token

# База данных документов
documents = [
    "Эверест — самая высокая гора в мире, высота которой составляет 8848 метров.",
    "Гора Эверест расположена в Гималаях на границе Непала и Тибета.",
    "Вулкан Фудзи — это символ Японии, его высота составляет 3776 метров.",
    "Эверест также известен как Джомолунгма, его высота 8848,86 метров.",
    "Гора Эверест является одной из самых популярных гор для альпинистов, однако восхождение на неё является очень опасным.",
    "Многие люди погибают при попытке подняться на вершину Эвереста из-за экстремальных условий и недостатка кислорода."
]

# Генерация эмбеддингов для документов
doc_embeddings = model.encode(documents, convert_to_tensor = True)

# Функция поиска наиболее релевантных документов
def retrieve(query, k=3):
    query_embedding = model.encode(query, convert_to_tensor = True)
    # Вычисление сходства между запросом и каждым документом (косинусное сходство)
    scores = util.cos_sim(query_embedding, doc_embeddings)
    
    # Преобразуем результат в массив и сортируем
    scores = scores.cpu().detach().numpy()  # Преобразуем тензор в массив NumPy
    top_k_indices = scores.argsort()[0][-k:][::-1]  # Индексы с наибольшими сходствами
    return [documents[i] for i in top_k_indices]

# Генерация ответа с использованием GPT-2
def generate_answer(query, retrieved_docs):
    input_text = f"Вопрос: {query} Контекст: {' '.join(retrieved_docs)}"
    inputs = tokenizer_gen(input_text, return_tensors = "pt", padding = True, truncation = True)
    outputs = model_gen.generate(inputs["input_ids"], max_new_tokens = 100, num_beams = 4, temperature = 0.7, top_p = 0.9)
    return tokenizer_gen.decode(outputs[0], skip_special_tokens = True)

# Пример
query = "Какая высота Эвереста?"
retrieved_docs = retrieve(query)
answer = generate_answer(query, retrieved_docs)

print("Запрос:", query)
print("Ответ:", answer)
