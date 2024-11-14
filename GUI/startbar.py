from PyQt5 import QtWidgets
from gui import Ui_MainWindow  # импорт нашего сгенерированного файла
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from ML.mod_res import pol_mod,nb_model,logreq,Ada,GradientBoost,CLF
from ML.load_token import load_tokenizer
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtCore import QThread, pyqtSignal, QObject

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_2.clicked.connect(self.exit)
        self.ui.textBrowser_2.setVisible( False)
        self.ui.radioButton.setChecked(True)
        self.toke = load_tokenizer()
        self.ui.textBrowser.setText("Наивный байесовский классификатор - это простой вероятностный классификатор, основанный на применении теоремы Байеса с наивным предположением о независимости между признаками. Этот алгоритм основан на допущении о том, что наличие определенного признака в классе не связано с наличием других признаков.")

        # Подключаем обработчик сигнала toggled
      #  self.ui.radioButton.toggled.connect(self.onRadioToggled)
        self.ui.radioButton_2.toggled.connect(self.onRadioToggled)
        self.ui.radioButton_3.toggled.connect(self.onRadioToggled)
        self.ui.radioButton_4.toggled.connect(self.onRadioToggled)
        self.ui.radioButton_5.toggled.connect(self.onRadioToggled)
        self.ui.radioButton_6.toggled.connect(self.onRadioToggled)
#        self.ui.radioButton_7.toggled.connect(self.onRadioToggled)
        self.ui.pushButton.clicked.connect(self.onPushToggled)
    def onPushToggled(self):
        self.thread = QThread()
        print('Анализ')
        if self.ui.textEdit.toPlainText() =="":
            self.ui.textBrowser_2.setText("Введите текст в окно выше")
            self.ui.textBrowser_2.setVisible(True)
            return
        if self.ui.radioButton_6.isChecked():
            self.pol=pol_mod(self.ui.textEdit.toPlainText())
            #self.pol.moveToThread(self.thread)
            #self.thread.started.connect(self.pol.start)
            self.ui.textBrowser_2.setText(str(self.pol.start()))
            self.ui.textBrowser_2.setVisible(True)

            #self.ui.textBrowser_2.setText()
        if self.ui.radioButton.isChecked():
            self.ui.textBrowser_2.setVisible(True)
            self.ui.textBrowser_2.setText(str("Ожидайте"))
            self.ui.textBrowser_2.setText(nb_model.nb_analiz(self.ui.textEdit.toPlainText()))
        if self.ui.radioButton_2.isChecked():
            self.ui.textBrowser_2.setVisible(True)
            self.ui.textBrowser_2.setText(str("Ожидайте"))
            self.ui.textBrowser_2.setText(logreq.nb_analiz(self.ui.textEdit.toPlainText()))
        if self.ui.radioButton_4.isChecked():
            self.ui.textBrowser_2.setVisible(True)
            self.ui.textBrowser_2.setText(str("Ожидайте"))
            self.ui.textBrowser_2.setText(Ada.nb_analiz(self.ui.textEdit.toPlainText()))
        if self.ui.radioButton_3.isChecked():
            self.ui.textBrowser_2.setVisible(True)
            self.ui.textBrowser_2.setText(str("Ожидайте"))
            self.ui.textBrowser_2.setText(CLF.nb_analiz(self.ui.textEdit.toPlainText()))
        if self.ui.radioButton_5.isChecked():
            self.ui.textBrowser_2.setVisible(True)
            self.ui.textBrowser_2.setText(str("Ожидайте"))
            self.ui.textBrowser_2.setText(GradientBoost.nb_analiz(self.ui.textEdit.toPlainText()))
    def exit(self):
        exit()
    def onRadioToggled(self):
        # Получаем объект, который инициировал сигнал
        radioButton = self.sender()

        if radioButton.isChecked():
            print(f'Выбрана радиокнопка: {radioButton.text()}')
            if radioButton.text() ==' наивный классификатор Байеса':
                self.ui.textBrowser.setText("Наивный байесовский классификатор - это простой вероятностный классификатор, основанный на применении теоремы Байеса с наивным предположением о независимости между признаками. Этот алгоритм основан на допущении о том, что наличие определенного признака в классе не связано с наличием других признаков.")
            if radioButton.text() == 'Логистическая регрессия':
                self.ui.textBrowser.setText("Логистическая регрессия. Также является простейшим алгоритмом классификации. С помощью данного алгоритма можно разделить несложные объекты на 2 класса. Модель логистической регрессии быстро обучается и подходит для задач бинарной классификации.")
            if radioButton.text() == 'Метод опорных векторов.':
                self.ui.textBrowser.setText("Метод опорных векторов (SVM) - это алгоритм машинного обучения, который строит гиперплоскость в n-мерном пространстве для оптимального разделения образцов разных классов, максимизируя расстояние между этой гиперплоскостью и ближайшими к ней образцами, называемыми опорными векторами, и использует ядерные функции для построения нелинейных разделяющих поверхностей в исходном признаковом пространстве, что делает его эффективным инструментом для классификации и регрессии в различных областях, хотя он чувствителен к выбору ядра и медленно работает на больших наборах данных.")
            if radioButton.text() == 'AdaBoost':
                self.ui.textBrowser.setText("AdaBoost (Adaptive Boosting) - это алгоритм ансамблевого машинного обучения, который последовательно строит слабые ученики (например, решающие деревья) и комбинирует их в сильный классификатор, фокусируясь на объектах, которые были неправильно классифицированы предыдущими учениками, и веса которых корректируются для улучшения общей производительности модели, что делает его эффективным методом для улучшения качества классификации на основе усиления слабых учеников.")
            if radioButton.text() == 'Градиентный бустинг':
                self.ui.textBrowser.setText("Градиентный бустинг - это метод ансамблевого машинного обучения, который строит последовательность слабых моделей (например, деревьев решений) итеративно, минимизируя функцию потерь с использованием градиентного спуска, при этом каждая новая модель обучается на ошибках предыдущих, что позволяет создавать сильные модели, способные к обработке разнообразных данных и решению различных задач, таких как классификация и регрессия, в том числе XGBoost, LightGBM и CatBoost являются популярными реализациями этого метода.")
            if radioButton.text() == 'Полносвязнная модель':
                self.ui.textBrowser.setText("Полносвязная модель (Fully Connected Model) - это архитектура искусственных нейронных сетей, в которой каждый нейрон в одном слое соединен с каждым нейроном в следующем слое, позволяя модели изучать сложные нелинейные зависимости в данных; также известна как многослойный персептрон (Multilayer Perceptron, MLP); эта модель часто используется для решения задач классификации и регрессии в областях компьютерного зрения, обработки естественного языка и других областях.")
            if radioButton.text() == 'Сверточная модель':
                self.ui.textBrowser.setText("Сверточная модель (Convolutional Neural Network, CNN) - это класс искусственных нейронных сетей, специально разработанных для анализа данных с пространственной структурой, такой как изображения; она состоит из нескольких слоев, включая сверточные слои для извлечения признаков из входных данных, слои подвыборки для уменьшения размерности и избежания переобучения, а также полносвязные слои для классификации или регрессии; CNN показывает выдающуюся производительность в задачах компьютерного зрения, распознавания образов и других задач, требующих анализа пространственных данных.")
app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())