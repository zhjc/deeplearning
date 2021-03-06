# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\chinn\Documents\QtProject\pyui\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("FaceIdentification")
        MainWindow.resize(779, 473)
        MainWindow.setStyleSheet("")
        MainWindow.setWindowIcon(QIcon('icons/net.ico'))
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.camera = QtWidgets.QLabel(self.centralWidget)
        self.camera.setGeometry(QtCore.QRect(30, 40, 341, 361))
        self.camera.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.camera.setText("")
        self.camera.setObjectName("camera")
        self.identification = QtWidgets.QPushButton(self.centralWidget)
        self.identification.setGeometry(QtCore.QRect(400, 180, 111, 31))
        self.identification.setStyleSheet("font: 11pt \"微软雅黑\";\n"
"color: rgb(0, 0, 255);\n"
"background-color: rgb(183, 183, 255);")
        self.identification.setObjectName("identification")
        self.top_k_label = QtWidgets.QLabel(self.centralWidget)
        self.top_k_label.setGeometry(QtCore.QRect(600, 10, 103, 21))
        self.top_k_label.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"color: rgb(0, 0, 255);")
        self.top_k_label.setObjectName("top_k_label")
        self.top4 = QtWidgets.QLabel(self.centralWidget)
        self.top4.setGeometry(QtCore.QRect(540, 300, 91, 91))
        self.top4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.top4.setText("")
        self.top4.setObjectName("top4")
        self.top2 = QtWidgets.QLabel(self.centralWidget)
        self.top2.setGeometry(QtCore.QRect(540, 170, 91, 91))
        self.top2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.top2.setText("")
        self.top2.setObjectName("top2")
        self.top1 = QtWidgets.QLabel(self.centralWidget)
        self.top1.setGeometry(QtCore.QRect(600, 40, 91, 91))
        self.top1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.top1.setText("")
        self.top1.setObjectName("top1")
        self.top3 = QtWidgets.QLabel(self.centralWidget)
        self.top3.setGeometry(QtCore.QRect(660, 170, 91, 91))
        self.top3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.top3.setText("")
        self.top3.setObjectName("top3")
        self.top5 = QtWidgets.QLabel(self.centralWidget)
        self.top5.setGeometry(QtCore.QRect(660, 300, 91, 91))
        self.top5.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.top5.setText("")
        self.top5.setObjectName("top5")
        self.top4_name = QtWidgets.QLabel(self.centralWidget)
        self.top4_name.setGeometry(QtCore.QRect(540, 390, 91, 21))
        self.top4_name.setStyleSheet("")
        self.top4_name.setText("")
        self.top4_name.setObjectName("top4_name")
        self.top5_name = QtWidgets.QLabel(self.centralWidget)
        self.top5_name.setGeometry(QtCore.QRect(660, 390, 91, 21))
        self.top5_name.setStyleSheet("")
        self.top5_name.setText("")
        self.top5_name.setObjectName("top5_name")
        self.top3_name = QtWidgets.QLabel(self.centralWidget)
        self.top3_name.setGeometry(QtCore.QRect(660, 260, 91, 21))
        self.top3_name.setStyleSheet("")
        self.top3_name.setText("")
        self.top3_name.setObjectName("top3_name")
        self.top2_name = QtWidgets.QLabel(self.centralWidget)
        self.top2_name.setGeometry(QtCore.QRect(540, 260, 91, 21))
        self.top2_name.setStyleSheet("")
        self.top2_name.setText("")
        self.top2_name.setObjectName("top2_name")
        self.top1_name = QtWidgets.QLabel(self.centralWidget)
        self.top1_name.setGeometry(QtCore.QRect(600, 130, 91, 21))
        self.top1_name.setStyleSheet("")
        self.top1_name.setText("")
        self.top1_name.setObjectName("top1_name")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 779, 23))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionOpencv_Align = QtWidgets.QAction(MainWindow)
        self.actionOpencv_Align.setObjectName("actionOpencv_Align")
        self.actionFlandmark_Align = QtWidgets.QAction(MainWindow)
        self.actionFlandmark_Align.setObjectName("actionFlandmark_Align")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionStart = QtWidgets.QAction(MainWindow)
        self.actionStart.setObjectName("actionStart")
        self.actionStop = QtWidgets.QAction(MainWindow)
        self.actionStop.setObjectName("actionStop")
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionOpencv_Align)
        self.menuFile.addAction(self.actionFlandmark_Align)
        self.menuFile.addAction(self.actionExit)
        self.menuEdit.addAction(self.actionStart)
        self.menuEdit.addAction(self.actionStop)
        self.menuHelp.addAction(self.actionAbout)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())
        self.menuBar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FaceIdentification"))
        self.identification.setText(_translate("MainWindow", "Identification"))
        self.top_k_label.setText(_translate("MainWindow", "Top-k Results"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Camera"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionOpencv_Align.setText(_translate("MainWindow", "OpencvA"))
        self.actionFlandmark_Align.setText(_translate("MainWindow", "FlandmarkA"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionStart.setText(_translate("MainWindow", "Start"))
        self.actionStop.setText(_translate("MainWindow", "Stop"))

