from PyQt5 import QtWidgets, QtMultimedia, QtMultimediaWidgets
import sys

class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__()

        # window settings
        self.setWindowTitle("QRScanner App")

        # main layout
        self.lay = QtWidgets.QVBoxLayout()

        # widgets
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.capture_button.clicked.connect(self.btn_click)
        self.label = QtWidgets.QLabel("")

        # setting the device
        self.device = QtMultimedia.QCamera.availableDevices()[0]
        self.m_camera = QtMultimedia.QCamera(self.device)

        #test
        self.label.setText(str(self.m_camera.availableDevices()))

        self.view_finder = QtMultimediaWidgets.QCameraViewfinder()
        self.view_finder.setMinimumSize(250, 250)

        self.m_camera.setViewfinder(self.view_finder)
        self.m_camera.setCaptureMode(QtMultimedia.QCamera.CaptureStillImage)
        self.m_camera.start()

        self.lay.addWidget(self.label)
        self.lay.addWidget(self.view_finder)
        self.lay.addWidget(self.capture_button)
        self.lay.addWidget(self.label)
        self.setLayout(self.lay)

    def btn_click(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())