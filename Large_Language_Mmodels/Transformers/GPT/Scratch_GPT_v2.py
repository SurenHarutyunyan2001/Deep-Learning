import tensorflow as tf
from keras.api.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout, Input
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences
from transformers import GPT2Tokenizer
import numpy as np

# Загрузка токенизатора GPT2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Устанавливаем токен для padding
tokenizer.pad_token = tokenizer.eos_token  # Используем токен конца последовательности как токен для padding

class GPT1(Model):
    def __init__(self, vocab_size, seq_len, d_model, num_heads, ff_dim, num_layers, dropout_rate=0.1):
        super(GPT1, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Эмбеддинги токенов
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=d_model)
        
        # Позиционные кодировки
        self.position_encoding = self.compute_position_encoding(seq_len, d_model)

        # Слои трансформера
        self.transformer_blocks = [
            self.build_transformer_block(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]

        # Dropout
        self.dropout_layer = Dropout(dropout_rate)

        # Финальный слой
        self.final_layer = Dense(vocab_size)

    def compute_position_encoding(self, seq_len, d_model):
        """
        Вычисляет позиционные кодировки для заданной длины последовательности и размерности модели.
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        # Применяем sin к четным индексам и cos к нечетным
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    def create_attention_mask(self, seq_len, batch_size):
        """
        Создание каузальной маски для внимания, чтобы скрыть будущие токены.
        Маска расширяется до формы (batch_size, seq_len, seq_len).
        """
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # (seq_len, seq_len)
        mask = tf.expand_dims(mask, axis=0)  # (1, seq_len, seq_len)
        mask = tf.tile(mask, [batch_size, 1, 1])  # (batch_size, seq_len, seq_len)
        return mask


    def build_transformer_block(self, d_model, num_heads, ff_dim, dropout_rate):
        inputs = Input(shape=(None, d_model))
        mask = Input(shape=(None, None))

        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs, attention_mask=mask)
        attention_output = Dropout(dropout_rate)(attention_output)
        norm_attention = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        ffn_output = Dense(ff_dim, activation='relu')(norm_attention)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)

        outputs = LayerNormalization(epsilon=1e-6)(norm_attention + ffn_output)
        return Model([inputs, mask], outputs)

    def call(self, inputs):
        """
        Основная логика вызова модели.
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Эмбеддинги токенов
        token_embeddings = self.token_embedding(inputs)

        # Позиционные кодировки (ограничиваем до текущей длины входов)
        position_embeddings = self.position_encoding[:seq_len, :]

        # Суммируем токеновые и позиционные эмбеддинги
        x = token_embeddings + position_embeddings

        # Создаём маску внимания
        attention_mask = self.create_attention_mask(seq_len, batch_size)

        # Применяем блоки трансформера
        for transformer_block in self.transformer_blocks:
            x = transformer_block([x, attention_mask])

        # Dropout
        x = self.dropout_layer(x)

        # Финальный линейный слой
        output = self.final_layer(x)

        return output

# Загрузка и подготовка данных IMDB
max_len = 100  # Максимальная длина последовательности

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

# Токенизация с помощью GPT2Tokenizer
x_train = tokenizer.batch_encode_plus(
    [' '.join([str(word) for word in review]) for review in x_train],
    padding = 'max_length',
    truncation = True,
    max_length = max_len,
    return_tensors = 'tf'
)['input_ids']

x_test = tokenizer.batch_encode_plus(
    [' '.join([str(word) for word in review]) for review in x_test],
    padding = 'max_length',
    truncation = True,
    max_length = max_len,
    return_tensors = 'tf'
)['input_ids']

# Сдвиг целевых данных для предсказания следующего токена
y_train = tf.concat([x_train[:, 1:], tf.fill([tf.shape(x_train)[0], 1], tokenizer.eos_token_id)], axis=1)
y_test = tf.concat([x_test[:, 1:], tf.fill([tf.shape(x_test)[0], 1], tokenizer.eos_token_id)], axis=1)

# Гиперпараметры
vocab_size = tokenizer.vocab_size  # Используем vocab_size токенизатора GPT-2
seq_len = max_len
d_model = 128
num_heads = 4
ff_dim = 512
num_layers = 4
dropout_rate = 0.1

# Создание модели
gpt1 = GPT1(vocab_size, seq_len, d_model, num_heads, ff_dim, num_layers, dropout_rate)

# Компиляция модели
gpt1.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

# Обучение модели
gpt1.fit(
    x_train, y_train,
    batch_size = 64,
    epochs = 5,
    validation_data = (x_test, y_test)
)

# Генерация текста с подсказкой
def generate_text_with_prompt(prompt, model, tokenizer, max_len = 100):
    input_tokens = tokenizer.encode(prompt, return_tensors = "tf")
    generated_sequence = input_tokens[0].numpy().tolist()

    for _ in range(max_len - len(generated_sequence)):
        padded_input = tf.pad(tf.convert_to_tensor([generated_sequence]), [[0, 0], [0, max_len - len(generated_sequence)]], "CONSTANT")
        predictions = model(padded_input)
        
        next_token = tf.argmax(predictions[:, len(generated_sequence) - 1, :], axis = -1).numpy()[0]
        generated_sequence.append(next_token)

        if next_token == tokenizer.eos_token_id:  # Если токен <EOS>, прекращаем генерацию
            break

    # Преобразуем сгенерированные токены обратно в текст
    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens = True)
    return generated_text

# Пример подсказки
prompt = "In a faraway land, there was a kingdom ruled by a wise king."
generated_text = generate_text_with_prompt(prompt, gpt1, tokenizer, max_len = 100)

print("Generated Text:", generated_text)
