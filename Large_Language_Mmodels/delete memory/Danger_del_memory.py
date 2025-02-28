import os
import shutil

def clear_cache_directory(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, followlinks=True):
            for dir_name in dirs:
                # Очистка всех подкаталогов в указанной папке
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path, ignore_errors=True)
                print(f"Кэш {dir_path} очищен.")

# Путь к кэшированным данным для различных библиотек (обычно ~/.cache или C:\Users\<user>\AppData\Local)
cache_dirs = [
    os.path.expanduser("~/.cache"),  # Для Linux/Mac
    os.path.expanduser("C:/Users/Tigran/AppData/Local")  # Для Windows
]

for cache_dir in cache_dirs:
    clear_cache_directory(cache_dir)
