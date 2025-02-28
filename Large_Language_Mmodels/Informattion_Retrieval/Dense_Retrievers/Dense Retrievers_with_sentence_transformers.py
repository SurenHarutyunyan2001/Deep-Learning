from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # Легковесная модель
documents = ["I love programming with Python", "Keras is a high-level API"]
query = "What is Keras?"

# Генерация эмбеддингов
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Вычисление сходства
scores = util.cos_sim(query_embedding, doc_embeddings)
print(scores)
