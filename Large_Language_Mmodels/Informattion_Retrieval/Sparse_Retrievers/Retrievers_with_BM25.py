from rank_bm25 import BM25Okapi
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

# Токенизируем документы и запросы
tokenized_documents = [doc.split() for doc in documents]
tokenized_queries = [query.split() for query in queries]

# Создаем модель BM25
bm25 = BM25Okapi(tokenized_documents)

# Для каждого запроса ищем наиболее релевантный документ
for query in tokenized_queries:
    scores = bm25.get_scores(query)
    most_relevant_doc_idx = np.argmax(scores)
    print(f"Query: {' '.join(query)}")
    print(f"Most relevant document: {documents[most_relevant_doc_idx]}")
    print(f"Score: {scores[most_relevant_doc_idx]}")
    print("=" * 50)
