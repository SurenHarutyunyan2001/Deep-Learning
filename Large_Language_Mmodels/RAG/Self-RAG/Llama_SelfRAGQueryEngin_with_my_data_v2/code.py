# Инструкция по скачиванию и установке пакетов с GitHub:
#https://github.com/run-llama/llama_index/tree/v0.10.20/llama-index-packs/llama-index-packs-self-rag

import os
import pandas as pd
from PyPDF2 import PdfReader
from llama_index import VectorStoreIndex, Document
from self_rag_pack.base import SelfRAGQueryEngine
from openai import GPT3


# Функция для чтения текста из .txt файла
def read_txt_file(file_path):
    with open(file_path, "r", encoding = "utf-8") as file:
        return file.read()

# Функция для чтения текста из .csv файла
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return " ".join(df.iloc[:, 0].astype(str).tolist())  # Преобразуем все данные в строку

# Функция для чтения текста из .pdf файла
def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Используем '' если страница не может быть прочитана
    return text

# Функция для чтения текста из файлов разных типов
def read_files(file_paths):
    all_text = ""
    
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".txt":
            all_text += read_txt_file(file_path) + "\n"
        elif file_extension == ".csv":
            all_text += read_csv_file(file_path) + "\n"
        elif file_extension == ".pdf":
            all_text += read_pdf_file(file_path) + "\n"
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    return all_text

# Функция для подготовки документов для использования в retriever
def prepare_documents(file_paths):
    # Чтение данных из файлов
    text = read_files(file_paths)

    # Разбиваем текст на отдельные документы (например, по предложениям или параграфам)
    documents = [Document(text = doc) for doc in text.split("\n") if doc.strip()]

    # Создаем индекс документов
    return VectorStoreIndex.from_documents(documents)

# Подготовка документов
file_paths = ["data.txt"]

# Настройка retriever
retriever = VectorStoreIndex.from_documents(file_paths).as_retriever()

# Настройка GPT-3 с использованием API
model = GPT3(api_key = "your-api-key")

# Настройка SelfRAGQueryEngine с GPT-3 в качестве модели генерации
query_engine = SelfRAGQueryEngine(
    model_path = model,  # Здесь передаем экземпляр модели GPT-3
    retriever = retriever,
    verbose = True
)

# Выполнение запроса через SelfRAGQueryEngine
response = query_engine.query("Whot is RAG ?")

# Печатаем результат
print(response)
