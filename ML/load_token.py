import json
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import pickle
def load_tokenizer():
    # Получаем текущую директорию
    # Получаем путь к текущему исполняемому файлу
    script_path = os.path.realpath(__file__)

    # Получаем путь к директории, в которой находится исполняемый файл
    script_directory = os.path.dirname(script_path)

    with open(script_directory+"\\tokenizer (1).json", "r", encoding='utf-8') as json_file:
        tokenizer_json = json.load(json_file)
        tokenizer = tokenizer_from_json(json.dumps(tokenizer_json))
        return tokenizer
def load_tokenizer_7k():
    # Получаем текущую директорию
    # Получаем путь к текущему исполняемому файлу
    script_path = os.path.realpath(__file__)

    # Получаем путь к директории, в которой находится исполняемый файл
    script_directory = os.path.dirname(script_path)

    with open(script_directory+"\\tokenizer (3).json", "r", encoding='utf-8') as json_file:
        tokenizer_json = json.load(json_file)
        tokenizer = tokenizer_from_json(json.dumps(tokenizer_json))
        return tokenizer

def toke_text(text,toke):
    text_to_tokenize=text
    tokenized_text = toke.texts_to_sequences([text_to_tokenize])
    tokenized_text = toke.sequences_to_matrix(tokenized_text)
    print(tokenized_text)
    return tokenized_text
def load_list():
    # Получаем текущую директорию
    # Получаем путь к текущему исполняемому файлу
    script_path = os.path.realpath(__file__)

    # Получаем путь к директории, в которой находится исполняемый файл
    script_directory = os.path.dirname(script_path)
    filename=script_directory+"\\className.pkl"
    with open(filename, 'rb') as f:
        # Загружаем список из файла
        my_list = pickle.load(f)
    return my_list