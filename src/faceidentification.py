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
        self.actionOpen_File.triggered.connect(self.openfile)
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
        
    def openfile(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Select File", "C:/", "jpg Files (*.jpg);;png Files (*.png);;")
        if filename is None or filename=="":
            return
        img = QImage(filename)
        if img is None:
            QMessageBox.warning(self, "Warning", self.tr("图片载入失败，请重新操作"))
            return
        else:
            qpm = QPixmap.fromImage(img)
            miniqpm = qpm.scaled(341,361,QtCore.Qt.KeepAspectRatio)
            self.camera.setPixmap(miniqpm)   

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    
    sys.exit(app.exec_())