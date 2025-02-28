from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

dataset = load_dataset("imdb")  
train_dataset = dataset["train"]
test_dataset = dataset["test"]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation = True)

train_dataset = train_dataset.map(tokenize_function, batched = True)
test_dataset = test_dataset.map(tokenize_function, batched = True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)  # для бинарной классификации

training_args = TrainingArguments(
    output_dir = "./results",          # где сохранять результаты
    num_train_epochs = 3,              # количество эпох
    per_device_train_batch_size = 8,   # размер батча
    per_device_eval_batch_size = 16,   # размер батча для теста
    warmup_steps = 500,                # количество шагов для разогрева
    weight_decay = 0.01,               # весовой распад
    logging_dir = "./logs",            # директория для логов
    logging_steps = 10,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)

trainer.train()

results = trainer.evaluate()
print(results)

def classify_review(review):
    inputs = tokenizer(review, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()  # индекс класса с наибольшим логитом
    return predicted_class

# Пример использования
review = "This product is amazing!"
print(classify_review(review))

