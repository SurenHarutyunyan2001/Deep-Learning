import sys
import re
sys.stdout.reconfigure(encoding = 'utf-8')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os
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

# Загружаем модель T5 для генерации текста
gen_model = pipeline(
    "text2text-generation",
    model = "t5-small",
    device = "cpu",
    max_length = 200,
    pad_token_id = 50256
)
llm = HuggingFacePipeline(pipeline = gen_model)

# Создаем RAG-пайплайн с обновленным Retriever
qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = vector_db.as_retriever())

# Тестируем систему
query = "What is RAG?"
response = qa_chain.invoke(query)

# Очистка ответа от лишних фрагментов
def clean_response(response):
    response_text = response["result"]
    cleaned_text = re.sub(r"\b(\w +)( \1) + \b", r"\1", response_text)
    cleaned_text = cleaned_text.replace("RAG", "")
    return cleaned_text.strip()

cleaned_response = clean_response(response)

print("Cleaned Answer:", cleaned_response)
#print("Answer: ", response)


