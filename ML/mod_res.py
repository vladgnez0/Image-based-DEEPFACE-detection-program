from tensorflow.keras.models import load_model
from ML.load_token import load_tokenizer,toke_text,load_list,load_tokenizer_7k
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import os
import numpy as np
import joblib
class pol_mod():
    finished = pyqtSignal()
    def __init__(self,text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\pol_mod.h5"
        self.model = load_model(path)  # Укажите путь к файлу вашей модели
        self.toke=load_tokenizer_7k()
        self.text=text

    def start(self):
        a= toke_text(self.text,self.toke)
        print(a.shape[1])
        # Обрезка данных, если они длиннее 7000
        if a.shape[1] > 7000:
            a = a[:, :7000]
        # Дополнение данных, если они короче 7000
        elif a.shape[1] < 7000:
            a = np.pad(a, ((0, 0), (0, 7000 - a.shape[1])), mode='constant', constant_values=0)
        result = self.model.predict(a)  # Замените input_data на ваши входные данные
        print(result)
        className=load_list()
        recognizedClass = np.argmax(result)
        print(recognizedClass,className)

        return  "Текст написан:", className[recognizedClass], "с вероятностью", result[0][recognizedClass]
class nb_model():
    finished = pyqtSignal()
    @staticmethod
    def nb_analiz(text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\naive_bayes_model.pkl"
        model = joblib.load(path)  # Укажите путь к файлу вашей модели
        toke=load_tokenizer()
        try:
            #a = np.argmax(toke_text(text, toke),axis=1)
            res=model.predict(toke_text(text, toke))
            className = load_list()
            recognizedClass = np.argmax(res)
            print(recognizedClass, className)

            return "Текст написан:"+ str(className[res[0]])
        except Exception as e:
            print(e)
            res=None
        return str(res)

class logreq:
    finished = pyqtSignal()
    @staticmethod
    def nb_analiz(text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\logreq.pkl"
        model = joblib.load(path)  # Укажите путь к файлу вашей модели
        toke=load_tokenizer()
        try:
            #a = np.argmax(toke_text(text, toke),axis=1)
            res=model.predict(toke_text(text, toke))
            className = load_list()
            recognizedClass = np.argmax(res)
            print(recognizedClass, className)

            return "Текст написан:"+str(className[res[0]])
        except Exception as e:
            print(e)
class Ada:
    finished = pyqtSignal()
    @staticmethod
    def nb_analiz(text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\Ada.pkl"
        try:
            model = joblib.load(path)  # Укажите путь к файлу вашей модели
            toke = load_tokenizer()
            #a = np.argmax(toke_text(text, toke),axis=1)
            res=model.predict(toke_text(text, toke))
            className = load_list()
            recognizedClass = np.argmax(res)
            print(recognizedClass, className)

            return "Текст написан:"+ str(className[res[0]])
        except Exception as e:
            print(e)
            return None
class GradientBoost:
    finished = pyqtSignal()
    @staticmethod
    def nb_analiz(text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\GradientBoostingClassifier.pkl"
        try:
            model = joblib.load(path)  # Укажите путь к файлу вашей модели
            toke = load_tokenizer()
            #a = np.argmax(toke_text(text, toke),axis=1)
            res=model.predict(toke_text(text, toke))
            className = load_list()
            print(res)
            #recognizedClass = np.argmax(res[0])
           # print(recognizedClass, className)

            return "Текст написан:"+ str(className[res[0]])
        except Exception as e:
            print(e)
            return None
class CLF:
    finished = pyqtSignal()
    @staticmethod
    def nb_analiz(text):
        # Получаем путь к текущему исполняемому файлу
        script_path = os.path.realpath(__file__)
        # Получаем путь к директории, в которой находится исполняемый файл
        script_directory = os.path.dirname(script_path)
        path= script_directory +"\\model\\CLF.pkl"
        try:
            model = joblib.load(path)  # Укажите путь к файлу вашей модели
            toke = load_tokenizer()
            #a = np.argmax(toke_text(text, toke),axis=1)
            res=model.predict(toke_text(text, toke))
            className = load_list()
            print(res)
            #recognizedClass = np.argmax(res[0])
           # print(recognizedClass, className)

            return "Текст написан:"+ str(className[res[0]])
        except Exception as e:
            print(e)
            return None