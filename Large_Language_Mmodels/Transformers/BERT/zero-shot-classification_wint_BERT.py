from transformers import pipeline

# Загрузка Zero-shot классификатора
classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")

# Пример текста для классификации
text = ["I love this product! It works great and is very reliable.","I dont like it"]

# Метки классов
candidate_labels = ["positive", "negative", "neutral"]

# Классификация
result = classifier(text, candidate_labels)

print(result)

# Вывод только метки с наибольшей вероятностью
for res in result:
    # Находим метку с максимальной вероятностью
    label = res['labels'][0]
    print(f"Classified as: {label}")