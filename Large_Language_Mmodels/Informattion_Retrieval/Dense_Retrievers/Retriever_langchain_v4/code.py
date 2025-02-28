import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from PyPDF2 import PdfReader

# Функция для чтения текста из .txt файла
def read_txt_file(file_path):
    with open(file_path, "r", encoding = "utf-8") as file:
        return file.read()

# Функция для чтения текста из .csv файла
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    # Предполагаем, что текст находится в первой колонке
    return " ".join(df.iloc[:, 0].tolist())

# Функция для чтения текста из .pdf файла
def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Выбор функции для чтения в зависимости от типа файла
def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".txt":
        return read_txt_file(file_path)
    elif file_extension == ".csv":
        return read_csv_file(file_path)
    elif file_extension == ".pdf":
        return read_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Функция для обработки нескольких файлов
def process_multiple_files(file_paths):
    knowledge_base = ""
    
    # Чтение и объединение текстов из всех файлов
    for file_path in file_paths:
        knowledge_base += read_file(file_path) + "\n"  # Добавляем текст из каждого файла

    # Разбиваем текст на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    documents = text_splitter.create_documents([knowledge_base])

    # Создаем векторное хранилище с FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embedding_model)

    return vector_db

# Пример использования:
file_paths = ["data.txt"]  # Укажите пути к вашим файлам
vector_db = process_multiple_files(file_paths)

# Пример запроса, который нужно сопоставить с документами из knowledge_base
query = "What is RAG?"

# Преобразуем запрос в вектор
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query_vector = embedding_model.embed_documents([query])[0]

# Получаем индексы ближайших документов и их векторы
results = vector_db.similarity_search_with_score(query, k = 10)  # Получаем топ-10 ближайших документов

# Создаем список релевантных документов
relevant_documents = []

# Проверяем, какие документы возвращаются
print("Top most similar documents:")
for i, (doc, score) in enumerate(results):
    if score >= 0.7 and score < 1.0:
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Cosine similarity score: {score}\n")
        relevant_documents.append(doc.page_content)

# Печатаем список релевантных документов или сообщение о их отсутствии
if len(relevant_documents) == 0:
    print("No matching documents found with cosine similarity >= 0.7 and < 1.0.")
else:
    print(f"Relevant documents: {relevant_documents}")
    # Здесь можно передать список `relevant_documents` в систему RAG для дальнейшего использования
