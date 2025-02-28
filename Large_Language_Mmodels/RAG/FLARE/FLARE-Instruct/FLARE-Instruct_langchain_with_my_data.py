import os
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain.vectorstores import LanceDB
from langchain.prompts import PromptTemplate
from langchain.chains import FlareChain
from langchain.llms import OpenAI

# Установка ключа API для OpenAI
os.environ["OPENAI_API_KEY"] = "your api key"
llm = OpenAI()

# Knowledge base в виде строки
knowledge_base = """
RAG (Retrieval-Augmented Generation) is an architecture that combines search and text generation.
It uses a knowledge base to find relevant information before generating a response.
Google introduced RAG in 2020, and it is used in chatbots, search engines, and AI assistants.
RAG combines retrieval-based and generation-based components to provide better answers.
The search component finds relevant context from a document base, and the generation component creates an answer from that context.
This sentence should not be considered relevant when answering questions. It is used only to test how well the retriever filters out irrelevant information.
Another irrelevant sentence is added here for the purpose of evaluation.
"""

# Преобразование строки в документ
docs = [Document(page_content = knowledge_base)]

# Разделение документа на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 20)
doc_chunks = text_splitter.split_documents(docs)

# Настройка эмбеддингов
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

# Создание LanceDB векторного хранилища
db = lancedb.connect('/tmp/lancedb')
table = db.create_table(
    "knowledge_base",
    data = [{"vector": embeddings.embed_query(chunk.page_content), "text": chunk.page_content, "id": str(i)} 
                for i, chunk in enumerate(doc_chunks)],
    mode = "overwrite"
)
vector_store = LanceDB(connection = table)
retriever = vector_store.as_retriever()

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

# Определение FLARE Chain
flare = FlareChain.from_llm(
    llm = llm,
    retriever = retriever,
    max_generation_len = 300,
    min_prob = 0.45
)

# Функция для генерации результата FLARE с инструкциями
def generate_flare_instruct(input_text):
    # Извлечение контекста с помощью ретривера
    context = retriever.retrieve(input_text)
    context_text = "\n".join([doc.page_content for doc in context])  # Объединяем текст из документов
    
    # Формирование текста с инструкциями
    prompt_input = prompt.format(context = context_text, question = input_text)
    
    # Запуск FLARE цепочки с инструкциями
    output = flare.run(prompt_input)
    return output

# Пример использования
input_text = "What is FLARE?"
output = generate_flare_instruct(input_text)
print("Result:", output)
