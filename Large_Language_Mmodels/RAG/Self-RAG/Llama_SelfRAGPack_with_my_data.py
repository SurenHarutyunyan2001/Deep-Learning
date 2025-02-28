# Инструкция по скачиванию и установке пакетов с GitHub:
#https://github.com/run-llama/llama_index/tree/v0.10.20/llama-index-packs/llama-index-packs-self-rag


from llama_index import VectorStoreIndex, Document
from openai import GPT3
from llm_compiler_agent_pack.base import SelfRAGPack

# Подготовка данных для поиска
documents = [
    Document("RAG (Retrieval-Augmented Generation) is an architecture that combines search and text generation."),
    Document("It uses a knowledge base to find relevant information before generating a response."),
    Document("Google introduced RAG in 2020, and it is used in chatbots, search engines, and AI assistants."),
    Document("RAG combines retrieval-based and generation-based components to provide better answers."),
    Document("The search component finds relevant context from a document base, and the generation component creates an answer from that context."),
    Document("This sentence should not be considered relevant when answering questions. It is used only to test how well the retriever filters out irrelevant information."),
    Document("Another irrelevant sentence is added here for the purpose of evaluation."),
]

# Настройка retriever
retriever = VectorStoreIndex.from_documents(documents).as_retriever()

# Настройка GPT-3 с использованием API
model = GPT3(api_key="your-api-key")

# Настройка SelfRAGPack с GPT-3 в качестве модели генерации
agent_pack = SelfRAGPack(
    model_path=model,  # Здесь передаем экземпляр модели GPT-3
    retriever=retriever,
    verbose=True
)

# Выполнение запроса через SelfRAGPack
response = agent_pack.query("What is RAG?")

# Печатаем результат
print(response)
