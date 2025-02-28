import sys
import re
sys.stdout.reconfigure(encoding = 'utf-8')

import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, T5Tokenizer

# Переведенный набор данных (knowledge base) на английский
knowledge_base = """
RAG (Retrieval-Augmented Generation) is an architecture that combines search and text generation.
It uses a knowledge base to find relevant information before generating a response.
Google introduced RAG in 2020, and it is used in chatbots, search engines, and AI assistants.
RAG combines retrieval-based and generation-based components to provide better answers.
The search component finds relevant context from a document base, and the generation component creates an answer from that context.
"""

# Дополнительные "не нужные" предложения
additional_irrelevant_info = """
This sentence should not be considered relevant when answering questions. It is used only to test how well the retriever filters out irrelevant information.
Another irrelevant sentence is added here for the purpose of evaluation.
"""

# Объединяем текст
knowledge_base += additional_irrelevant_info

# Разбиваем текст на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
documents = text_splitter.create_documents([knowledge_base])

# Создаем векторное хранилище с FAISS
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embedding_model)

# Загружаем модель T5 для генерации текста и токенизатор
gen_model = pipeline(
    "text2text-generation",
    model = "t5-small",  # Используем модель t5-small 
    device = "cpu",  # Указываем устройство (cpu или cuda)
    max_length = 200,  # Увеличиваем количество новых токенов
    pad_token_id = 50256  # Указываем pad_token_id для T5
)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Создаем RAG-Token пайплайн
def rag_token_pipeline(query, vector_db):
    # Токенизируем запрос
    query_tokens = tokenizer.encode(query, return_tensors = "pt")
    
    # Ищем релевантные документы
    retrieved_docs = vector_db.similarity_search(query, k = 1)  # Извлекаем один документ
    
    # Токенизируем контекст из документа
    doc_text = retrieved_docs[0].page_content
    doc_tokens = tokenizer.encode(doc_text, return_tensors = "pt")
    
    # Объединяем токены запроса и документа
    input_tokens = torch.cat((query_tokens, doc_tokens), dim = 1)
    
    # Генерация ответа на основе токенов
    generated_tokens = gen_model.model.generate(input_tokens, max_length = 200, pad_token_id = 50256)
    
    # Декодируем и возвращаем результат
    return tokenizer.decode(generated_tokens[0], skip_special_tokens = True)

# Тестируем систему с RAG-Token
query = "What is RAG?"
response = rag_token_pipeline(query, vector_db)

# Очистка ответа от лишних фрагментов
def clean_response(response):
    # Убираем повторяющиеся фразы и "шум"
    cleaned_text = re.sub(r"\b(\w+)( \1)+\b", r"\1", response)  # Убираем дубликаты слов
    cleaned_text = cleaned_text.replace("RAG", "")  # Убираем слова или фразы, которые могут быть неуместными
    return cleaned_text.strip()

cleaned_response = clean_response(response)

print("Cleaned Answer:", cleaned_response)
