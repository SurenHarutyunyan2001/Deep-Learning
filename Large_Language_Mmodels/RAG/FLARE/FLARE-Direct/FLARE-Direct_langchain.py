from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import ArxivLoader
from langchain.vectorstores import LanceDB
from langchain.chains import FlareChain
from langchain.llms import OpenAI
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

# Определение FLARE Chain
flare = FlareChain.from_llm(
    llm = llm,
    retriever = vector_store_retriever,
    max_generation_len = 300,
    min_prob = 0.45
)

# Функция для генерации результата FLARE
def generate_flare_output(input_text):
    output = flare.run(input_text)
    return output


# Пример 
input_text = "What is FLARE?"
output = generate_flare_output(input_text)
print("Result:", output)
