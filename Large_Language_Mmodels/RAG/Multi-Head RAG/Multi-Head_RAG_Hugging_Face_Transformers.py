import torch
import os
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Отключаем предупреждение про символические ссылки
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# инициализация токенизатора и модели
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever_1 = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name = "exact")
retriever_2 = RagRetriever.from_pretrained("facebook/rag-token-wiki", index_name = "exact")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever = [retriever_1, retriever_2])

# пример пользовательского вопроса
question = "What are the recent advancements in quantum computing?"
input_ids = tokenizer(question, return_tensors = "pt").input_ids

# извлечение релевантных документов из нескольких источников
retrieved_docs_1 = retriever_1(input_ids)
retrieved_docs_2 = retriever_2(input_ids)
combined_docs = retrieved_docs_1 + retrieved_docs_2

# генерация ответа
outputs = model.generate(input_ids, context_input_ids = combined_docs, num_beams = 5, num_return_sequences = 1)
generated_answer = tokenizer.decode(outputs[0], skip_special_tokens = True)

print("Question:", question)
print("Answer:", generated_answer)