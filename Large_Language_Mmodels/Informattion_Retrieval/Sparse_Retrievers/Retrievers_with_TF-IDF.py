from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Пример коллекции документов
documents = [
    "I love programming with Python",
    "Keras is a high-level neural networks API",
    "Transformers are powerful for NLP tasks",
    "Deep learning is the future of AI"
]

# Запросы
queries = [
    "I enjoy coding in Python",
    "What is Keras?",
    "Tell me about Transformers",
    "Future of AI"
]

# 1. Создаем TF-IDF векторизатор
vectorizer = TfidfVectorizer()

# Обучаем векторизатор на документах
tfidf_matrix = vectorizer.fit_transform(documents)

# 2. Преобразуем запросы в разреженные векторы
query_vectors = vectorizer.transform(queries)


# 3. Сравниваем запросы с документами с использованием cosine_similarity
results = []
for i, query_vector in enumerate(query_vectors):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_relevant_doc_idx = np.argmax(cosine_similarities)
    results.append((queries[i], documents[most_relevant_doc_idx], cosine_similarities[most_relevant_doc_idx]))

# 4. Выводим результаты
for query, document, similarity in results:
    print(f"Query: {query}")
    print(f"Most relevant document: {document}")
    print(f"Cosine Similarity: {similarity}")
    print("=" * 50)

    
