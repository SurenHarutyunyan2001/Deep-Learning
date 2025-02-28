from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from trl import PPOTrainer, PPOConfig, DPOTrainer, DPOConfig
from trl import RewardModel, HumanFeedbackDataset
from trl import FeedbackConfig

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
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Загружаем датасет IMDb и применяем токенизацию
dataset_imdb = load_dataset("imdb")  # Загружаем датасет IMDb
train_imdb = dataset_imdb["train"].map(tokenize_function, batched=True)  # Токенизируем тренировочные данные
test_imdb = dataset_imdb["test"].map(tokenize_function, batched=True)  # Токенизируем тестовые данные

# Настроим параметры RLHF
feedback_config = FeedbackConfig(
    beta = 0.1,  # Коэффициент обучения для RL
    learning_rate = 1e-6,
    batch_size = 8,
    gradient_accumulation_steps = 2,
    human_feedback_type = "pairwise",  # Используем парный тип обратной связи
    reward_model = "ppo"  # Тип модели для вознаграждения (можно настроить для других типов)
)

# Подготовка датасета с человеческой обратной связью
feedback_dataset = HumanFeedbackDataset(
    train_dataset = train_imdb, 
    eval_dataset = test_imdb, 
    tokenizer = tokenizer,
    feedback_type = "pairwise",  # Парный тип обратной связи
)

# Инициализируем модель вознаграждения (Reward Model)
reward_model = RewardModel(
    model = model,
    tokenizer = tokenizer,
    feedback_type = "pairwise"
)

# Инициализируем тренера для RLHF с использованием PPO
trainer = PPOTrainer(
    model = model,
    reward_model = reward_model,
    train_dataset = feedback_dataset,
    eval_dataset = test_imdb,
    tokenizer = tokenizer,
    config = feedback_config,
)

# Запускаем обучение
trainer.train()

# Сохраняем модель после обучения
model.save_pretrained("bert-rlhf-imdb")  # Сохраняем модель в указанную папку

# Оценка модели на тестовых данных
results = trainer.evaluate()  # Оценка модели на тестовых данных
print(results)  # Выводим результаты

# Функция для классификации отзыва с использованием модели после RLHF
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
