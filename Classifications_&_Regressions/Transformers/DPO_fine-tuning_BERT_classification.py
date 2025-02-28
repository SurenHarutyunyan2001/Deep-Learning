from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from trl import DPOTrainer, DPOConfig

# Загружаем модель BERT для классификации
model_name = "bert-base-uncased"  # Имя модели BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Загружаем токенизатор для модели BERT
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 2,  # Для задачи бинарной классификации
    torch_dtype = torch.float16,  # Используем float16 для более быстрой работы с памятью
).to("cuda" if torch.cuda.is_available() else "cpu")  # Загружаем модель на GPU, если доступно

# Функция для токенизации текста
def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True, max_length = 512)  # Токенизируем текст с максимальной длиной 512 токенов

# Загружаем датасет IMDb и применяем токенизацию
dataset_imdb = load_dataset("imdb")  # Загружаем датасет IMDb
train_imdb = dataset_imdb["train"].map(tokenize_function, batched = True)  # Токенизируем тренировочные данные
test_imdb = dataset_imdb["test"].map(tokenize_function, batched = True)  # Токенизируем тестовые данные

# Настроим параметры DPO
config = DPOConfig(
    beta = 0.1,  # Коэффициент обучения
    learning_rate = 1e-6,
    batch_size = 8,
    gradient_accumulation_steps = 2
)

# Настроим параметры тренировки
training_args = TrainingArguments(
    output_dir = "./results",  # Папка для сохранения результатов
    evaluation_strategy = "epoch",  # Оценка модели после каждой эпохи
    learning_rate = 1e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,  # Количество эпох
    weight_decay = 0.01,  # Вес для регуляризации
    logging_dir = "./logs",  # Папка для логов
    logging_steps = 10,
)

# 4️⃣ Запускаем тренер
trainer = DPOTrainer(
    model = model,
    args = training_args,
    train_dataset = train_imdb,
    eval_dataset = test_imdb,
    tokenizer = tokenizer,
    config = config
)
trainer.train()  # Начинаем обучение

# Сохраняем модель после обучения
model.save_pretrained("bert-lora-imdb")  # Сохраняем модель в указанную папку

# Оценка модели на тестовых данных
results = trainer.evaluate()  # Оценка модели на тестовых данных
print(results)  # Выводим результаты

# Функция для классификации отзыва
def classify_review(review):
    # Токенизируем отзыв
    inputs = tokenizer(review, 
                       return_tensors = "pt",
                       padding = True, 
                       truncation = True, 
                       max_length = 512).to(model.device)  # Убедитесь, что данные на том же устройстве, что и модель
    
    # Прогоняем токенизированный отзыв через модель
    with torch.no_grad():  # Отключаем градиенты для инференса
        outputs = model(**inputs)
    logits = outputs.logits  # Логиты модели
    predicted_class = logits.argmax().item()  # Индекс класса с наибольшим логитом
    return predicted_class  # Возвращаем предсказанный класс

# Пример использования функции для классификации отзыва
review = "This product is amazing!"  # Пример отзыва
print(classify_review(review))  # Выводим предсказанный класс
