from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Загружаем модель BERT для классификации
model_name = "bert-base-uncased"  # Имя модели BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Загружаем токенизатор для модели BERT
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2).to("cuda" if torch.cuda.is_available() else "cpu")  # Загружаем модель для классификации с двумя метками (положительный/отрицательный)

# Функция для токенизации текста
def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True, max_length = 512)  # Токенизируем текст с максимальной длиной 512 токенов

# Загружаем датасет IMDb и применяем токенизацию
dataset_imdb = load_dataset("imdb")  # Загружаем датасет IMDb
train_imdb = dataset_imdb["train"].map(tokenize_function, batched = True)  # Токенизируем тренировочные данные
test_imdb = dataset_imdb["test"].map(tokenize_function, batched = True)  # Токенизируем тестовые данные

# Настройки LoRA для адаптеров
lora_config = LoraConfig(
    r = 8,  # Размер адаптера
    lora_alpha = 32,  # Альфа для LoRA
    lora_dropout = 0.1,  # Дроп-аут для LoRA
    target_modules = ["attention.self.query", 
                      "attention.self.key", 
                      "attention.self.value", 
                      "attention.output.dense"]  # Модули, к которым применяется LoRA
)

# Применяем LoRA к модели BERT
model = get_peft_model(model, lora_config)  # Применяем LoRA-адаптеры к модели

# Аргументы для тренировки
training_args = TrainingArguments(
    output_dir = "./results_imdb",  # Папка для сохранения результатов
    num_train_epochs = 3,  # Количество эпох для обучения
    per_device_train_batch_size = 8,  # Размер батча для обучения
    per_device_eval_batch_size = 16,  # Размер батча для оценки
    warmup_steps = 500,  # Количество шагов для разогрева
    weight_decay = 0.01,  # Параметр веса для регуляризации
    logging_dir = "./logs",  # Папка для логов
    logging_steps = 10,  # Частота логирования
    evaluation_strategy = "epoch",  # Стратегия для оценки модели (по окончании каждой эпохи)
    save_strategy = "epoch"  # Стратегия для сохранения модели (по окончании каждой эпохи)
)

# Создаём тренера и запускаем обучение
trainer = Trainer(
    model = model,  # Модель, которую обучаем
    args = training_args,  # Аргументы для тренировки
    train_dataset = train_imdb,  # Тренировочные данные
    eval_dataset = test_imdb  # Тестовые данные
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
                       max_length = 512)
    
    # Прогоняем токенизированный отзыв через модель
    outputs = model(**inputs)
    logits = outputs.logits  # Логиты модели
    predicted_class = logits.argmax().item()  # Индекс класса с наибольшим логитом
    return predicted_class  # Возвращаем предсказанный класс

# Пример использования функции для классификации отзыва
review = "This product is amazing!"  # Пример отзыва
print(classify_review(review))  # Выводим предсказанный класс

""""
Значение	    Где используется?	        Описание
"q_proj"	    Все модели	                Query projection (внимание)
"k_proj"	    Все модели	                Key projection (внимание)
"v_proj"	    Все модели	                Value projection (внимание)
"o_proj"	    Все модели	                Output projection (внимание)
"fc1"	        BERT, T5	                Первый слой FFN
"fc2"	        BERT, T5	                Второй слой FFN
"gate_proj"	    LLaMA	                    Управление FFN
"up_proj"	    LLaMA	                    Увеличение размерности в FFN
"down_proj"	    LLaMA	                    Уменьшение размерности в FFN
"lm_head"	    GPT, LLaMA	                Выходной слой (логиты слов)
"""