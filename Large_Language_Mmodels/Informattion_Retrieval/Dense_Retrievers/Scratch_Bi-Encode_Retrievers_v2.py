from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация модели BERT и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Пример данных
documents = [
    "I love programming with Python",
    "Keras is a high-level neural networks API",
    "Transformers are powerful for NLP tasks",
    "Deep learning is the future of AI"
]

queries = [
    "I enjoy coding in Python",
    "What is Keras?",
    "Tell me about Transformers in NLP",
    "AI and Deep Learning advancements"
]

# Функция для получения векторных представлений с использованием BERT
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, padding = True, truncation = True, return_tensors = 'tf')
    outputs = bert_model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Используем [CLS] токен как представление
    return embeddings

# Получаем векторные представления для документов и запросов
document_embeddings = get_bert_embeddings(documents)
query_embeddings = get_bert_embeddings(queries)

# Вычисляем сходство (косинусное расстояние)
similarities = cosine_similarity(query_embeddings.numpy(), document_embeddings.numpy())

# Выводим наиболее релевантные документы для каждого запроса
for i, query in enumerate(queries):
    relevant_document_idx = similarities[i].argmax()
    print(f"Query: {query}")
    print(f"Most relevant document: {documents[relevant_document_idx]}")
    print(f"Similarity score: {similarities[i][relevant_document_idx]}")
    print("=" *50)
