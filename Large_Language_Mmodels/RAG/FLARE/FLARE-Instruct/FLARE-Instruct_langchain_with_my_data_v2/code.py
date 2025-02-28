import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import FlareChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import pandas as pd

# Установите ключ OpenAI API
os.environ["OPENAI_API_KEY"] = "your api key"

# Чтение текстового файла
def read_txt_file(file_path):
    with open(file_path, "r", encoding = "utf-8") as file:
        return file.read()

# Чтение CSV файла
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return " ".join(df.iloc[:, 0].tolist())

# Чтение PDF файла
def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Функция для чтения файла любого поддерживаемого типа
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

# Обработка нескольких файлов
def process_multiple_files(file_paths):
    knowledge_base = ""
    for file_path in file_paths:
        knowledge_base += read_file(file_path) + "\n"

    # Разбиваем текст на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    documents = text_splitter.create_documents([knowledge_base])

    # Создаем векторное хранилище с LanceDB
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    # Подключаем LanceDB
    db = LanceDB("/tmp/lancedb")
    table = db.create_table(
        "knowledge_base",
        data = [{"vector": embeddings.embed_query(doc.page_content), "text": doc.page_content, "id": str(i)}
            for i, doc in enumerate(documents)],
        mode = "overwrite"
    )
    vector_db = LanceDB(connection = table)
    return vector_db

# Загрузка данных из файлов
file_paths = ["data.txt"]  
vector_db = process_multiple_files(file_paths)

# Настройка FLARE Chain с инструкциями
llm = OpenAI()

# Шаблон для инструкций
prompt_template = """
You are a helpful assistant. Your task is to answer the following question using the context provided below.

Context:
{context}

Question:
{question}

Answer: 
"""

# Создание PromptTemplate с динамическими вставками
prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = prompt_template
)

flare = FlareChain.from_llm(
    llm = llm,
    retriever = vector_db.as_retriever(),
    max_generation_len = 300,
    min_prob = 0.45
)

# Функция для генерации результата FLARE с инструкциями
def generate_flare_instruct(input_text):
    # Извлечение контекста с помощью ретривера
    context = vector_db.as_retriever().retrieve(input_text)
    context_text = "\n".join([doc.page_content for doc in context])  # Объединяем текст из документов
    
    # Формирование текста с инструкциями
    prompt_input = prompt.format(context = context_text, question = input_text)
    
    # Запуск FLARE цепочки с инструкциями
    output = flare.run(prompt_input)
    return output

# Очистка ответа
def clean_response(response):
    cleaned_text = re.sub(r"\b(\w+)( \1)+\b", r"\1", response)
    return cleaned_text.strip()

# Пример использования
query = "What is RAG?"
response = generate_flare_instruct(query)
cleaned_response = clean_response(response)
print("Cleaned Answer:", cleaned_response)
