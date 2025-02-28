import os
import shutil
import tempfile
import gc
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Очистка неиспользуемых объектов
gc.collect()

# Установите список каталогов, которые должны быть очищены
cache_dirs = [
    os.path.expanduser("~/.cache/huggingface/transformers"),
    os.path.expanduser("~/.cache/huggingface/datasets"),
    os.path.expanduser("~/.cache/huggingface/tokenizers"),
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.expanduser("~/.keras"),
    os.path.expanduser("~/.tensorflow_hub"),
    os.path.expanduser("~/.cache/torch"),
    os.path.expanduser("~/.cache/langchain"),
    os.path.expanduser("~/.cache/sentence-transformers"),
    os.path.expanduser("~/.cache/spacy"),
    os.path.expanduser("~/.cache/scikit-learn"),
    os.path.expanduser("~/.cache/pip"),
    os.path.expanduser("~/.cache/joblib"),
    os.path.expanduser("~/.cache/faiss"),
    os.path.expanduser("~/.cache/farm-haystack"),
    os.path.expanduser("~/.cache/pandas")
]

# Функция для безопасного удаления
def safe_remove(directory):
    if os.path.exists(directory) and not any(
        part in directory for part in ["Visual Studio", "VS2019", "Code", "Program Files"]
    ):
        shutil.rmtree(directory)
        print(f"Кэш в {directory} очищен.")

# Удаляем только безопасные кэши
for cache_dir in cache_dirs:
    safe_remove(cache_dir)

# Очистка временных файлов, исключая папки, используемые системными программами
tempdir = tempfile.gettempdir()
for filename in os.listdir(tempdir):
    file_path = os.path.join(tempdir, filename)
    try:
        if os.path.isfile(file_path) and not any(part in file_path for part in ["Visual Studio", "VS2019", "Code"]):
            os.remove(file_path)
    except Exception as e:
        print(f"Не удалось удалить {file_path}: {e}")
