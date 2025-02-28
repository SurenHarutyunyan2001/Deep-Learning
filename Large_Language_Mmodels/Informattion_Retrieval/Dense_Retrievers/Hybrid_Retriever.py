from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch

# Проверяем, есть ли GPU
if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


# Мы используем Bi-Encoder для кодирования всех пассажа, чтобы использовать его для семантического поиска
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     # Усечение длинных пассажа до 256 токенов
top_k = 32                          # Количество пассажа, которые мы хотим получить с помощью bi-encoder

# Bi-encoder будет извлекать 100 документов. Мы используем Cross-encoder для повторной оценки списка результатов для улучшения качества
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# В качестве датасета используется Simple English Wikipedia. В отличие от полной английской Википедии, в ней только
# около 170 тысяч статей. Мы разделяем эти статьи на параграфы и кодируем их с помощью bi-encoder

wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

# Если файл не существует, загружаем его
if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())

        # Добавляем все параграфы
        #passages.extend(data['paragraphs'])

        # Добавляем только первый параграф
        passages.append(data['paragraphs'][0])

print("Passages:", len(passages))

# Мы кодируем все пассажа в наше векторное пространство. Это займет примерно 5 минут (в зависимости от скорости вашего GPU)
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

# Мы также сравниваем результаты с лексическим поиском (поиск по ключевым словам). Здесь мы используем
# алгоритм BM25, который реализован в пакете rank_bm25.

# Приводим текст к нижнему регистру и удаляем стоп-слова из индексации
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


tokenized_corpus = []
for passage in tqdm(passages):
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

# Эта функция будет искать все статьи Википедии по пассажам, которые отвечают на запрос
def search(query):
    print("Input question:", query)

    ##### Поиск с использованием BM25 (лексический поиск) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print("Top-3 результатов лексического поиска (BM25)")
    for hit in bm25_hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    ##### Семантический поиск #####
    # Кодируем запрос с помощью bi-encoder и находим потенциально релевантные пассажа
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Получаем результаты для первого запроса

    ##### Переоценка #####
    # Теперь оцениваем все извлеченные пассажа с помощью cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Сортируем результаты по оценкам от cross-encoder
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Выводим топ-3 результатов из bi-encoder
    print("\n-------------------------\n")
    print("Top-3 результата извлечения с помощью Bi-Encoder")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Выводим топ-3 результата из переоценки с помощью cross-encoder
    print("\n-------------------------\n")
    print("Top-3 результата после переоценки с помощью Cross-Encoder")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
        
# Пример использования функции поиска
search(query = "What is the capital of the United States?")
