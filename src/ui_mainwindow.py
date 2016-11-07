# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\chinn\Documents\QtProject\main_window\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ImageWorks(object):
    def setupUi(self, ImageWorks):
        ImageWorks.setObjectName("ImageWorks")
        ImageWorks.resize(759, 441)
        self.centralWidget = QtWidgets.QWidget(ImageWorks)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 90, 61, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 160, 61, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(350, 230, 61, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_5.setGeometry(QtCore.QRect(350, 300, 61, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(440, 20, 291, 361))
        self.label.setStyleSheet("selection-color: rgb(170, 170, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 291, 361))
        self.label_2.setStyleSheet("selection-color: rgb(170, 170, 255);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        ImageWorks.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(ImageWorks)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 759, 23))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.menuAbout = QtWidgets.QMenu(self.menuBar)
        self.menuAbout.setObjectName("menuAbout")
        ImageWorks.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(ImageWorks)
        self.mainToolBar.setObjectName("mainToolBar")
        ImageWorks.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(ImageWorks)
        self.statusBar.setObjectName("statusBar")
        ImageWorks.setStatusBar(self.statusBar)
        self.actionExit = QtWidgets.QAction(ImageWorks)
        self.actionExit.setObjectName("actionExit")
        self.actionAbout = QtWidgets.QAction(ImageWorks)
        self.actionAbout.setObjectName("actionAbout")
        self.actionOpen_File = QtWidgets.QAction(ImageWorks)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.menu.addAction(self.actionOpen_File)
        self.menu.addAction(self.actionExit)
        self.menuAbout.addAction(self.actionAbout)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(ImageWorks)
        self.actionExit.triggered.connect(ImageWorks.close)
        QtCore.QMetaObject.connectSlotsByName(ImageWorks)

    def retranslateUi(self, ImageWorks):
        _translate = QtCore.QCoreApplication.translate
        ImageWorks.setWindowTitle(_translate("ImageWorks", "ImageWorks"))
        self.pushButton_2.setText(_translate("ImageWorks", "图片分类"))
        self.pushButton_3.setText(_translate("ImageWorks", "人脸定位"))
        self.pushButton_4.setText(_translate("ImageWorks", "人脸识别"))
        self.pushButton_5.setText(_translate("ImageWorks", "在线识别"))
        self.menu.setTitle(_translate("ImageWorks", "File"))
        self.menuAbout.setTitle(_translate("ImageWorks", "Help"))
        self.actionExit.setText(_translate("ImageWorks", "Exit"))
        self.actionAbout.setText(_translate("ImageWorks", "About"))
        self.actionOpen_File.setText(_translate("ImageWorks", "Open File"))

