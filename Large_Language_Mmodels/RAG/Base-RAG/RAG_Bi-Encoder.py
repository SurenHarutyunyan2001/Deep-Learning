import torch
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np

# Инициализация моделей и токенизаторов
query_encoder_model = BertModel.from_pretrained('bert-base-uncased')
query_encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

doc_encoder_model = BertModel.from_pretrained('bert-base-uncased')
doc_encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

generator_model = T5ForConditionalGeneration.from_pretrained('t5-base')
generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Перенос моделей на устройство
query_encoder_model.to(device)
doc_encoder_model.to(device)
generator_model.to(device)

# Функция для кодирования текста в вектор
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, padding = True, max_length = 512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim = 1).cpu().numpy()

# Кодирование документов
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Germany is known for its beer culture.",
    "Mount Everest is the highest mountain in the world."
]

document_embeddings = np.vstack([encode_text(doc, doc_encoder_tokenizer, doc_encoder_model) for doc in documents])

# Индексирование с помощью Faiss
index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Индекс с использованием L2 расстояния
index.add(document_embeddings)

# Кодирование запроса
query = "Where is the Eiffel Tower located?"
query_embedding = encode_text(query, query_encoder_tokenizer, query_encoder_model)

# Извлечение релевантных документов
k = 2  # Количество извлекаемых документов
distances, indices = index.search(query_embedding, k)

# Извлечение текстов документов
retrieved_docs = [documents[i] for i in indices[0]]

# Генерация ответа с использованием T5
context = " ".join(retrieved_docs)
input_text = f"question: {query} context: {context}"
inputs = generator_tokenizer(input_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 512).to(device)

# Генерация ответа
with torch.no_grad():
    output = generator_model.generate(inputs.input_ids, max_length = 100, num_beams = 5, early_stopping = True)

generated_answer = generator_tokenizer.decode(output[0], skip_special_tokens = True)

# Печать ответа
print("Generated Answer:", generated_answer)
