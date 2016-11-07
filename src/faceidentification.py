# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 22:36:47 2016

@author: chinn
"""

import sys, os

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize 
from ui_faceidentification import Ui_MainWindow

from PyQt5 import QtWidgets, QtMultimediaWidgets, QtCore
from PyQt5.QtMultimedia import QCamera,QCameraImageCapture

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.actionAbout.triggered.connect(self.about)
        self.actionStart.triggered.connect(self.start)
        self.actionStop.triggered.connect(self.stop)
        self.identification.clicked.connect(self.capture_and_idfy)
        
        self.startcamera = False
        
    def about(self):
        QMessageBox.about(self, "About the system", self.tr("图形识别系统"))
        
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def capture_and_idfy(self):
        pass
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    
    sys.exit(app.exec_())