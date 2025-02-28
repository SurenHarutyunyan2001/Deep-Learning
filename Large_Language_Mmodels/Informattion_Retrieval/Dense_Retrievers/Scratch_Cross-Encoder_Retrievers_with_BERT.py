import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import *
from transformers import BertTokenizer, TFBertModel

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Загружаем предобученную модель BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Пример данных: документы и запросы
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

# Метки релевантности: 1 - релевантный, 0 - нерелевантный
labels = np.array([
    [1, 0, 0, 0],  
    [0, 1, 0, 0],  
    [0, 0, 1, 0],  
    [0, 0, 0, 1]   
], dtype=np.float32)

# Функция токенизации
def text_to_sequence(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'np'
    )
    return np.array([encoding['input_ids'][0], encoding['attention_mask'][0]], dtype = np.int32)

# Определяем максимальную длину последовательности
max_len = 256

# Токенизируем и сразу преобразуем в np.array(dtype=np.int32)
document_sequences = np.array([
    text_to_sequence(doc, tokenizer, max_len) for doc in documents
], dtype = np.int32)

query_sequences = np.array([
    text_to_sequence(query, tokenizer, max_len) for query in queries
], dtype = np.int32)

# Разделяем индексы и маски внимания
document_ids, document_attention_masks = document_sequences[:, 0, :], document_sequences[:, 1, :]
query_ids, query_attention_masks = query_sequences[:, 0, :], query_sequences[:, 1, :]


# Создание модели для объединенных запросов и документов
def create_cross_encoder_model():
    query_input = Input(shape = (max_len,), dtype = tf.int32)
    doc_input = Input(shape = (max_len,), dtype = tf.int32)

    # Объединяем запрос и документ в одну последовательность
    combined_input = Concatenate()([query_input, doc_input])

    # Маска внимания для комбинированной последовательности
    attention_mask = Concatenate()([query_attention_masks, document_attention_masks])

    # Пропускаем через BERT
    bert_output = bert_model(combined_input, attention_mask = attention_mask)[0]
    pooled_output = GlobalAveragePooling1D()(bert_output)

    # Выходной слой для классификации
    output = Dense(1, activation = 'sigmoid')(pooled_output)

    return Model(inputs=[query_input, doc_input], outputs = output)

# Создаем модель Cross-Encoder
cross_encoder_model = create_cross_encoder_model()

# Параметры обучения
learning_rate = 1e-5  
batch_size = 4
epochs = 10

# Функция потерь и оптимизатор
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# Компиляция модели
cross_encoder_model.compile(optimizer = optimizer, loss = loss_fn, metrics = ["accuracy"])

# Подготовка данных для обучения
queries_repeated = np.repeat(query_ids, len(documents), axis = 0)
documents_tiled = np.tile(document_ids, (len(queries), 1))
labels_flattened = labels.flatten()

# Маски внимания для комбинированных данных
attention_mask_queries = np.where(queries_repeated != 0, 1, 0)
attention_mask_docs = np.where(documents_tiled != 0, 1, 0)
combined_attention_mask = np.concatenate([attention_mask_queries, attention_mask_docs], axis = -1)

# Обучение модели
cross_encoder_model.fit(
    [queries_repeated, documents_tiled],
    labels_flattened,
    batch_size = batch_size,
    epochs = epochs
)

# Вычисляем сходство для всех запросов и документов
similarities = []

for query_seq, query_mask in zip(query_ids, query_attention_masks):
    query_similarities = []
    for doc_seq, doc_mask in zip(document_ids, document_attention_masks):
        # Мы передаем как объединенную последовательность запрос и документ
        combined_attention_mask = np.concatenate([query_mask[None, :], doc_mask[None, :]], axis = -1)
        similarity = cross_encoder_model.predict([query_seq[None, :], doc_seq[None, :]])
        query_similarities.append(similarity[0][0])  # Достаем значение предсказания
    similarities.append(query_similarities)

similarities = np.array(similarities)

# Выводим наиболее релевантные документы для каждого запроса
for i, query in enumerate(queries):
    relevant_doc_idx = np.argmax(similarities[i])
    print(f"Query: {query}")
    print(f"Most relevant document: {documents[relevant_doc_idx]}")
    print(f"Similarity score: {similarities[i][relevant_doc_idx]}")
    print("=" * 50)
