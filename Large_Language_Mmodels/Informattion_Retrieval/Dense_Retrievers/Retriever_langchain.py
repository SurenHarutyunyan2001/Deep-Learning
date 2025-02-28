import sys
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.stdout.reconfigure(encoding = 'utf-8')

# Переведенный набор данных (knowledge base) на английский
knowledge_base = """
RAG (Retrieval-Augmented Generation) is an architecture that combines search and text generation.
It uses a knowledge base to find relevant information before generating a response.
Google introduced RAG in 2020, and it is used in chatbots, search engines, and AI assistants.
RAG combines retrieval-based and generation-based components to provide better answers.
The search component finds relevant context from a document base, and the generation component creates an answer from that context.
"""

# Дополнительные "не нужные" предложения
additional_irrelevant_info = """
This sentence should not be considered relevant when answering questions. It is used only to test how well the retriever filters out irrelevant information.
Another irrelevant sentence is added here for the purpose of evaluation.
"""

# Объединяем текст
knowledge_base += additional_irrelevant_info

# Разбиваем текст на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
documents = text_splitter.create_documents([knowledge_base])

# Создаем векторное хранилище с FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embedding_model)

# Пример запроса, который нужно сопоставить с документами из knowledge_base
query = "What is RAG?"

# Преобразуем запрос в вектор
query_vector = embedding_model.embed_documents([query])[0]

# Получаем индексы ближайших документов и их векторы
results = vector_db.similarity_search_with_score(query, k=10)  # Получаем топ-10 ближайших документов

# Проверяем, какие документы возвращаются
print("Top 10 most similar documents:")
for i, (doc, score) in enumerate(results):
    # Нормализуем значение сходства (ограничиваем его до диапазона от 0 до 1)
    normalized_score = np.clip(score, 0, 1)
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Cosine similarity score: {normalized_score}\n")


