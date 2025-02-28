from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

# Загрузка данных из директории (предполагается, что в папке находятся документы)
documents = SimpleDirectoryReader("data.txt").load_data()

# Настройка параметров для разбиения текста: размер куска (512 токенов) и перекрытие между кусками (50 токенов)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Создание индекса на основе загруженных документов
index = VectorStoreIndex.from_documents(documents)

# Преобразование индекса в движок для выполнения запросов, ограничение на 4 наиболее похожих результата
query_engine = index.as_query_engine(similarity_top_k = 4)

# Выполнение запроса к индексу
response = query_engine.query("Ask something about your documents")

# Вывод полученного ответа
print(response)
