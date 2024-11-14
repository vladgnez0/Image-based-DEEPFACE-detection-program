
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation, SimpleRNN
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DatasetLoader:
    def __init__(self):
        # Загружаем обучающие тексты
        self.train_texts = []
        self.train_texts.append(self.read_text('content/(О. Генри) Обучающая_50 вместе.txt'))
        self.train_texts.append(self.read_text('content/(Стругацкие) Обучающая_5 вместе.txt'))
        self.train_texts.append(self.read_text('content/(Булгаков) Обучающая_5 вместе.txt'))
        self.train_texts.append(self.read_text('content/(Клиффорд_Саймак) Обучающая_5 вместе.txt'))
        self.train_texts.append(self.read_text('content/(Макс Фрай) Обучающая_5 вместе.txt'))
        self.train_texts.append(self.read_text('content/(Рэй Брэдберри) Обучающая_22 вместе.txt'))
        self.class_names = ["О. Генри", "Стругацкие", "Булгаков", "Саймак", "Фрай", "Брэдбери", ]
        self.num_classes = len(self.class_names)
        # Загружаем тестовые тексты
        self.test_texts = []
        self.test_texts.append(self.read_text('content/(О. Генри) Тестовая_20 вместе.txt'))
        self.test_texts.append(self.read_text('content/(Стругацкие) Тестовая_2 вместе.txt'))
        self.test_texts.append(self.read_text('content/(Булгаков) Тестовая_2 вместе.txt'))
        self.test_texts.append(self.read_text('content/(Клиффорд_Саймак) Тестовая_2 вместе.txt'))
        self.test_texts.append(self.read_text('content/(Макс Фрай) Тестовая_2 вместе.txt'))
        self.test_texts.append(self.read_text('content/(Рэй Брэдберри) Тестовая_8 вместе.txt'))
        # Смотрим размеры загруженных выборок
        for i in range(len(self.train_texts)):
            print("Длина обучающего текста", self.class_names[i], "\t", len(self.train_texts[i]), "\tПроверочного:", "\t",
                  len(self.test_texts[i]))

        self.maxWordsCount = 20000  # макс. кол-во слов/индексов для обучения текстов
        tokenizer = Tokenizer(num_words=self.maxWordsCount,
                              filters='!–"—#$%&amp;amp;()*+,-./:;&amp;lt;=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                              split=' ', char_level=False)
        tokenizer.fit_on_texts(self.train_texts)  # передаем тексты для получения токенов отсортированных по частоте повторяемости в количестве maxWordsCount
        self.items = list(tokenizer.word_index.items())  # берем индексы слов для просмотра
        print(self.items[:20])

        self.dist = list(tokenizer.word_counts.items())
        print(self.dist[:20])
        print(self.train_texts[0][:100])

        self.count_thres = 2  # кол-во раз меньше которого слово нужно исключить из списка
        self.low_count_words = [w for w, c in tokenizer.word_counts.items() if
                           c < self.count_thres]  # создаем список слов встречаюихся менее count_thres
        print(len(self.low_count_words))
        print(self.low_count_words[:20])
        for w in self.low_count_words:  # удаляем такие слова из исходного списка
            del tokenizer.word_index[w]
            del tokenizer.word_docs[w]
            del tokenizer.word_counts[w]
        self.trainWordIndexes = tokenizer.texts_to_sequences(self.train_texts)  # обучающие тесты в индексы
        self.testWordIndexes = tokenizer.texts_to_sequences(self.test_texts)  # проверочные тесты в индексы
        print("Исходный текст:\t\t", self.train_texts[1][:87])
        print("Он же в виде последовательности индексов:\t", self.trainWordIndexes[1][:20])
        self.symbs = 0
        self.words = 0
        for i in range(len(self.train_texts)):
            print(self.class_names[i], "\t", len(self.train_texts[i]), "символов,", len(self.trainWordIndexes[i]), " слов")
            self.symbs += len(self.train_texts[i])
            self.words += len(self.trainWordIndexes[i])

        print("\r\nВ сумме:\t", self.symbs, "символов,", self.words, "слов")
        print()
        self.arr = [x for x in range(23)]
        print("Длина:", len(self.arr), "\r\n")
        print("Входной текст:", self.arr, "\r\n")
        self.indexes = self.getSetFromIndexes(self.arr, 10, 3)
        for i in range(len(self.indexes)):
            print(self.indexes[i])
        print("Общий размер:", len(self.indexes), "x", len(self.indexes[0]), '=', len(self.indexes[0]) * len(self.indexes))
        arr = to_categorical(self.trainWordIndexes[0])
        print(len(self.trainWordIndexes[0]))
        print(self.arr.shape)
        print(self.trainWordIndexes[0][:10])
        print(self.arr[:20])
        self.texts = []
        self.texts.append([x for x in range(23)])  # Тестовый текст 1
        self.texts.append([x for x in range(100, 123)])  # Тестовый текст 2
        self.xTrain, self.yTrain = self.createSetsMultiClasses(self.texts, 10, 3)
        print(self.xTrain)
        print(self.yTrain)
        # Задаём базовые параметры
        self.xLen = 7000  # Длина отрезка текста в результирующемвекторе в словах
        self.shift = 100  # Смещение окна для разбиения исходного текста на обучающие вектора
        self.xTrain, self.yTrain = self.createSetsMultiClasses(self.trainWordIndexes, self.xLen, self.shift)  # Формируем обучающую выборку
        self.xTest, self.yTest = self.createSetsMultiClasses(self.testWordIndexes, self.xLen, self.shift)  # Формируем тестовую выборку
        print(self.xTrain.shape)
        print(self.yTrain.shape)
        print(self.xTest.shape)
        print(self.yTest.shape)
        self.xTrain01 = tokenizer.sequences_to_matrix(self.xTrain.tolist())  # Конвертируем xTrain в список перед передачей методу
        self.xTest01 = tokenizer.sequences_to_matrix(self.xTest.tolist())  # Конвертируем xTest в список перед передачей методу
        print(self.xTrain01.shape)  # Размер обучающей выборки, сформированной по Bag of Words
        print(self.xTrain01[0][100:120])  # фрагмент набора слов в виде Bag of Words
    def createSetsMultiClasses(self,wordIndexes, xLen, step):
        # Для каждого из классов создаём обучающую/проверочную выборку из индексов
        nClasses = len(wordIndexes)  # задаем количество классов выборки
        xSamples = []  # здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна(например 15779*1000)"
        ySamples = []  # здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной по количеству классов"
        for t, wI in enumerate(wordIndexes):
            tmp = self.getSetFromIndexes(wI, xLen, step)  # получаем список индексов, разбитый на "кол-во окон * длину окна"
            xSamples += tmp
            ySamples += [utils.to_categorical(t, nClasses).tolist()] * len(tmp)

        return (np.array(xSamples), np.array(ySamples))
    ###########################
    # Формирование обучающей выборки по листу индексов слов
    # (разделение на короткие векторы)
    ##########################
    def getSetFromIndexes(self,wordIndexes, xLen, shift):
        xSample = []
        wordsLen = len(wordIndexes)
        index = 0

        # Идём по всей длине вектора индексов
        # "Выбираем" блоки текст длины xLen и смещаемся вперёд на shift
        while (index + xLen <= wordsLen):
            xSample.append(wordIndexes[index:index + xLen])
            index += shift

        return xSample
    def replace_multiple(self, main_string, to_be_replaced, new_string):
        # Iterate over the strings to be replaced
        for elem in to_be_replaced:
            # Check if string is in the main string
            if elem in main_string:
                # Replace the string
                main_string = main_string.replace(elem, new_string)

        return main_string

    def read_text(self, file_name):  # функция принимает имя файла
        with open(file_name, 'r', encoding='utf-8') as file:  # открываем файла в режиме чтения
            text = file.read()  # читаем текст
            text = self.replace_multiple(text, ['\n\r', '\n', '\r', '\\xa0'], " ")  # заменяем переносы и спецсимволы разделителей на пробелы
        return text


a = DatasetLoader()