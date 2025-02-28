import os
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Отключаем предупреждение про символические ссылки
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Выбираем модель
model_name = "facebook/rag-sequence-base"

# Загружаем токенизатор
tokenizer = RagTokenizer.from_pretrained(model_name)

# Загружаем retriever с индексом для Википедии
retriever = RagRetriever.from_pretrained(model_name, index_name = "wiki")

# Загружаем RAG-модель
model = RagSequenceForGeneration.from_pretrained(model_name, retriever = retriever)

# Входной вопрос
question = "Who developed the theory of relativity?"

# Токенизируем ввод
input_ids = tokenizer(question, return_tensors = "pt").input_ids

# Генерация ответа
generated = model.generate(input_ids, max_length = 50, num_beams = 5, early_stopping = True)

# Декодируем результат
answer = tokenizer.batch_decode(generated, skip_special_tokens = True)[0]

print("Ответ:", answer)
