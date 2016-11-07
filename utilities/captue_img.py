# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PyQt5.QtMultimedia import QCamera,QCameraImageCapture
from PyQt5 import QtMultimediaWidgets

cm = QCamera()
viewfinder = QtMultimediaWidgets.QCameraViewfinder()
viewfinder.show()
cm.setViewfinder(viewfinder)

cp = QCameraImageCapture(cm)
cm.setCaptureMode(cm.CaptureStillImage)
cm.start()
cm.searchAndLock()

cp.capture()

cm.unlock()