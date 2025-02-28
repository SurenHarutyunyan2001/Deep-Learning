import os
from datasets import load_dataset
import shutil
import tempfile
import gc
import sys
sys.stdout.reconfigure(encoding = 'utf-8')

# Очистить неиспользуемые объекты
gc.collect()

gc.collect()


cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

datasets_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(datasets_cache_dir):
    shutil.rmtree(datasets_cache_dir)


tempdir = tempfile.gettempdir()
for filename in os.listdir(tempdir):
    file_path = os.path.join(tempdir, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Не удалось удалить {file_path}: {e}")


# Удалить кэш моделей
transformers_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(transformers_cache_dir):
    shutil.rmtree(transformers_cache_dir)
    print("Кэш моделей очищен.")

# Удалить кэш датасетов
datasets_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(datasets_cache_dir):
    shutil.rmtree(datasets_cache_dir)
    print("Кэш датасетов очищен.")

# Поиск всех папок, связанных с Hugging Face
cache_dirs = [
    os.path.expanduser("~/.cache/huggingface"),
    os.path.expanduser("~/.cache/transformers"),
    os.path.expanduser("~/.cache/datasets"),
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        print(f"Удаление кэша из {cache_dir}")
        shutil.rmtree(cache_dir)

torch_cache_dir = os.path.expanduser("~/.cache/torch")
if os.path.exists(torch_cache_dir):
    shutil.rmtree(torch_cache_dir)
    print("Кэш PyTorch очищен.")

hub_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hub_cache_dir):
    shutil.rmtree(hub_cache_dir)
    print("Кэш huggingface_hub очищен.")

# Очистка кэша Keras и TensorFlow
keras_cache_dir = os.path.expanduser("~/.keras")
if os.path.exists(keras_cache_dir):
    shutil.rmtree(keras_cache_dir)
    print("Кэш Keras очищен.")

tf_cache_dir = os.environ.get("TFHUB_CACHE", "~/.tensorflow_hub")
tf_cache_dir = os.path.expanduser(tf_cache_dir)
if os.path.exists(tf_cache_dir):
    shutil.rmtree(tf_cache_dir)
    print("Кэш TensorFlow очищен.")

# Очистка кэша haystack
haystack_cache_dir = os.path.expanduser("~/.haystack")
if os.path.exists(haystack_cache_dir):
    shutil.rmtree(haystack_cache_dir)
    print("Кэш Haystack очищен.")

# Очистка кэша sentence_transformers
sentence_transformers_cache = os.path.expanduser("~/.cache/sentence-transformers")
if os.path.exists(sentence_transformers_cache):
    shutil.rmtree(sentence_transformers_cache)
    print("Кэш Sentence-Transformers очищен.")

# Очистка кэша sklearn
sklearn_cache_dir = os.path.expanduser("~/.cache/scikit-learn")
if os.path.exists(sklearn_cache_dir):
    shutil.rmtree(sklearn_cache_dir)
    print("Кэш Scikit-learn очищен.")

# Очистка для Rank-BM25 (если кэш использовался)
rank_bm25_cache = "/path/to/rank_bm25_cache"  # Укажите путь, если известно
if os.path.exists(rank_bm25_cache):
    shutil.rmtree(rank_bm25_cache)
    print("Кэш Rank-BM25 очищен.")

# Очистка для langchain
langchain_cache_dir = os.path.expanduser("~/.cache/langchain")
if os.path.exists(langchain_cache_dir):
    shutil.rmtree(langchain_cache_dir)
    print("Кэш LangChain очищен.")

# Очистка для tokenizers
tokenizers_cache_dir = os.path.expanduser("~/.cache/huggingface/tokenizers")
if os.path.exists(tokenizers_cache_dir):
    shutil.rmtree(tokenizers_cache_dir)
    print("Кэш Tokenizers очищен.")

# Удаление всех JSON-файлов в определенной папке (если таковые есть)
json_cache_dir = "/path/to/your/json_cache"  # Укажите путь к кэшированным JSON-файлам
if os.path.exists(json_cache_dir):
    shutil.rmtree(json_cache_dir)
    print("Кэш JSON очищен.")

# Очистка для cmake 
cmake_cache_dir = os.path.expanduser("~/.cmake")
if os.path.exists(cmake_cache_dir):
    shutil.rmtree(cmake_cache_dir)
    print("Кэш CMake очищен.")
# Очистка для spacy
spacy_cache_dir = os.path.expanduser("~/.cache/spacy")
if os.path.exists(spacy_cache_dir):
    shutil.rmtree(spacy_cache_dir)
    print("Кэш spaCy очищен.")

# Очистка для huggingface_hub
huggingface_hub_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(huggingface_hub_cache_dir):
    shutil.rmtree(huggingface_hub_cache_dir)
    print("Кэш HuggingFace Hub очищен.")

# Очистка для pydantic
pydantic_cache_dir = "/path/to/your/pydantic_cache"  # Укажите путь, если таковой был использован
if os.path.exists(pydantic_cache_dir):
    shutil.rmtree(pydantic_cache_dir)
    print("Кэш Pydantic очищен.")

# Очистка для transformers
transformers_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(transformers_cache_dir):
    shutil.rmtree(transformers_cache_dir)
    print("Кэш Transformers очищен.")

# Очистка для torch
torch_cache_dir = os.path.expanduser("~/.cache/torch")
if os.path.exists(torch_cache_dir):
    shutil.rmtree(torch_cache_dir)
    print("Кэш Torch очищен.")

# Очистка для faiss
faiss_cache_dir = os.path.expanduser("~/.cache/faiss")
if os.path.exists(faiss_cache_dir):
    shutil.rmtree(faiss_cache_dir)
    print("Кэш Faiss очищен.")

# Очистка для farm_haystack
farm_haystack_cache_dir = os.path.expanduser("~/.cache/farm-haystack")
if os.path.exists(farm_haystack_cache_dir):
    shutil.rmtree(farm_haystack_cache_dir)
    print("Кэш Farm-Haystack очищен.")

# Очистка для numpy
numpy_cache_dir = os.path.expanduser("~/.cache/joblib")
if os.path.exists(numpy_cache_dir):
    shutil.rmtree(numpy_cache_dir)
    print("Кэш Numpy очищен.")

# Очистка для pandas
pandas_cache_dir = os.path.expanduser("~/.cache/pandas")
if os.path.exists(pandas_cache_dir):
    shutil.rmtree(pandas_cache_dir)
    print("Кэш Pandas очищен.")

# Очистка для pip
pip_cache_dir = os.path.expanduser("~/.cache/pip")
if os.path.exists(pip_cache_dir):
    shutil.rmtree(pip_cache_dir)
    print("Кэш Setuptools и Wheel очищен.")