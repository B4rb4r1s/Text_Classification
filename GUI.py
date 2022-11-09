import numpy as np
import random
from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType

from bert_dataset import CustomDataset
from bert_classifier import BertClassifier
from bert_prediction import predict

import torch
import torch.nn as nn


class Main(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        form, base = loadUiType('./GUI.ui')
        self.ui = form()
        self.ui.setupUi(self)

        self.ui.lineEdit.setText('На Украине пожаловались на болезненные удары армии российских дронов')

        self.ui.pushButton.clicked.connect(self.pred)

    def pred(self):
        token = './Models/rubert-tiny'
#         bert-tiny_1.0
        model = './Models/bert-tiny_1.0.pt'
        # text = str(self.ui.textEdit.toPlainText())
        text = str(self.ui.lineEdit.text())
        print(predict(token, model, text))
        # self.ui.label.setText(predict(token, model, text))
        # return text


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
