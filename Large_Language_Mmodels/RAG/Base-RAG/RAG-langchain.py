import openai
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

# Устанавливаем API-ключ для OpenAI
openai.api_key = "your api key"  # Замените на ваш правильный API-ключ

# Текстовые данные для индексации
text_data = [
    "RAG (Retrieval-Augmented Generation) is an architecture that combines search and text generation.",
    "It uses a knowledge base to find relevant information before generating a response.",
    "Google introduced RAG in 2020, and it is used in chatbots, search engines, and AI assistants.",
    "RAG combines retrieval-based and generation-based components to provide better answers.",
    "The search component finds relevant context from a document base, and the generation component creates an answer from that context.",
    "This sentence should not be considered relevant when answering questions. It is used only to test how well the retriever filters out irrelevant information.",
    "Another irrelevant sentence is added here for the purpose of evaluation.",
]

# Создание объектов для векторных представлений и поиска
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(text_data, embeddings)

# Настройка поискового механизма
retriever = vector_store.as_retriever()

# Создание цепочки для ответа на вопросы
qa_chain = RetrievalQA.from_chain_type(
    llm = OpenAI(temperature = 0),  # Используется OpenAI для генерации ответа
    retriever = retriever
)

# Запрос
query = "What is RAG ?"

# Получение ответа
response = qa_chain.invoke(query)

# Вывод ответа
print("Answer:", response.get("Result", response))  # Защита от возможной ошибки
