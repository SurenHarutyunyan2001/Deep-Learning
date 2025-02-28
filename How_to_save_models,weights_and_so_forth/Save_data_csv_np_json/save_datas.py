import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import numpy as np
import pandas as pd
import json
import os
from transformers import BertTokenizer

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Пример данных (документы и запросы)
documents = [
    "I love programming with Python",
    "Keras is a high-level neural networks API",
    "Transformers are powerful for NLP tasks",
    "Deep learning is the future of AI"
]

queries = [
    "I enjoy coding in Python",
    "What is Keras?",
    "Tell me about Transformers in NLP",
    "AI and Deep Learning advancements"
]

# Функция токенизации
def text_to_sequence(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'tf'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]

# Определим максимальную длину последовательности
max_len = 256

# Токенизация документов и запросов
document_sequences = [text_to_sequence(doc, tokenizer, max_len) for doc in documents]
query_sequences = [text_to_sequence(query, tokenizer, max_len) for query in queries]

# Разделяем индексы и маски внимания
document_ids, document_attention_masks = zip(*document_sequences)
query_ids, query_attention_masks = zip(*query_sequences)

# Преобразуем в numpy массивы
document_ids = np.array(document_ids, dtype = np.int32)
document_attention_masks = np.array(document_attention_masks, dtype = np.int32)
query_ids = np.array(query_ids, dtype = np.int32)
query_attention_masks = np.array(query_attention_masks, dtype = np.int32)

# Создаем папку для сохранения
save_dir = "./saved_data/"
os.makedirs(save_dir, exist_ok = True)

# Сохранение в NumPy (.npy)
np.save(os.path.join(save_dir, "document_ids.npy"), document_ids)
np.save(os.path.join(save_dir, "document_attention_masks.npy"), document_attention_masks)
np.save(os.path.join(save_dir, "query_ids.npy"), query_ids)
np.save(os.path.join(save_dir, "query_attention_masks.npy"), query_attention_masks)

# Сохранение в JSON (.json)
with open(os.path.join(save_dir, "document_ids.json"), "w") as f:
    json.dump(document_ids.tolist(), f)
with open(os.path.join(save_dir, "document_attention_masks.json"), "w") as f:
    json.dump(document_attention_masks.tolist(), f)
with open(os.path.join(save_dir, "query_ids.json"), "w") as f:
    json.dump(query_ids.tolist(), f)
with open(os.path.join(save_dir, "query_attention_masks.json"), "w") as f:
    json.dump(query_attention_masks.tolist(), f)

# Сохранение в CSV (.csv)
pd.DataFrame(document_ids).to_csv(os.path.join(save_dir, "document_ids.csv"), index = False, header = False)
pd.DataFrame(document_attention_masks).to_csv(os.path.join(save_dir, "document_attention_masks.csv"), index = False, header = False)
pd.DataFrame(query_ids).to_csv(os.path.join(save_dir, "query_ids.csv"), index = False, header = False)
pd.DataFrame(query_attention_masks).to_csv(os.path.join(save_dir, "query_attention_masks.csv"), index = False, header = False)

print("Файлы сохранены в:", save_dir)

# === ЗАГРУЗКА ДАННЫХ ===

# Загрузка NumPy
document_ids_loaded = np.load(os.path.join(save_dir, "document_ids.npy"))
document_attention_masks_loaded = np.load(os.path.join(save_dir, "document_attention_masks.npy"))
query_ids_loaded = np.load(os.path.join(save_dir, "query_ids.npy"))
query_attention_masks_loaded = np.load(os.path.join(save_dir, "query_attention_masks.npy"))

# Загрузка JSON
with open(os.path.join(save_dir, "document_ids.json"), "r") as f:
    document_ids_json_loaded = np.array(json.load(f), dtype = np.int32)
with open(os.path.join(save_dir, "document_attention_masks.json"), "r") as f:
    document_attention_masks_json_loaded = np.array(json.load(f), dtype = np.int32)
with open(os.path.join(save_dir, "query_ids.json"), "r") as f:
    query_ids_json_loaded = np.array(json.load(f), dtype = np.int32)
with open(os.path.join(save_dir, "query_attention_masks.json"), "r") as f:
    query_attention_masks_json_loaded = np.array(json.load(f), dtype = np.int32)

# Загрузка CSV
document_ids_csv_loaded = np.array(pd.read_csv(os.path.join(save_dir, "document_ids.csv"), header = None), dtype = np.int32)
document_attention_masks_csv_loaded = np.array(pd.read_csv(os.path.join(save_dir, "document_attention_masks.csv"), header = None), dtype = np.int32)
query_ids_csv_loaded = np.array(pd.read_csv(os.path.join(save_dir, "query_ids.csv"), header = None), dtype = np.int32)
query_attention_masks_csv_loaded = np.array(pd.read_csv(os.path.join(save_dir, "query_attention_masks.csv"), header = None), dtype = np.int32)

# === ПРОВЕРКА ===
print("Проверка загруженных данных:")
print("NumPy:", np.array_equal(document_ids, document_ids_loaded))
print("JSON:", np.array_equal(document_ids, document_ids_json_loaded))
print("CSV:", np.array_equal(document_ids, document_ids_csv_loaded))
