from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import ArxivLoader
from langchain.vectorstores import LanceDB
from langchain.chains import FlareChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import lancedb

# Установка ключа API для OpenAI
os.environ["OPENAI_API_KEY"] = "sk-your api key"
llm = OpenAI()

# Параметры для эмбеддингов HuggingFace BGE
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

# Загрузка документа с ArXiv (пример FLARE paper)
docs = ArxivLoader(query = "2305.06983", load_max_docs = 2).load()

# Разделение документа на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150)
doc_chunks = text_splitter.split_documents(docs)

# Создание LanceDB векторного хранилища
db = lancedb.connect('/tmp/lancedb')
table = db.create_table(
    "documentsai",
    data = [{"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}],
    mode = "overwrite"
)
vector_store = LanceDB.from_documents(doc_chunks, embeddings, connection = table)
vector_store_retriever = vector_store.as_retriever()

# Шаблон для инструкций (PromptTemplate)
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

# Создание FLARE Chain с инструкцией
flare = FlareChain.from_llm(
    llm = llm,
    retriever = vector_store_retriever,
    max_generation_len = 300,
    min_prob = 0.45
)

# Функция для генерации результата FLARE с инструкциями
def generate_flare_instruct(input_text):
    # Извлечение контекста с помощью ретривера
    context = vector_store_retriever.retrieve(input_text)
    
    # Формирование текста с инструкциями
    prompt_input = prompt.format(context = context, question = input_text)
    
    # Запуск FLARE цепочки с инструкциями
    output = flare.run(prompt_input)
    return output

# Пример использования
input_text = "What is FLARE?"
output = generate_flare_instruct(input_text)
print("Result:", output)
