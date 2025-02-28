import openai
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

# Устанавливаем API-ключ для OpenAI
openai.api_key = "your api key"  

# Чтение данных из файла
with open('data.txt', 'r', encoding = 'utf-8') as file:
    text_data = file.readlines()  # Читает все строки из файла и сохраняет их в список

# Убираем возможные символы новой строки
text_data = [line.strip() for line in text_data]

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
