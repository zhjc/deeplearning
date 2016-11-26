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

import cv2

import caffe
import skimage
import numpy as np

import sklearn.metrics.pairwise as pw

OPENCV_ALIGN = 0
FLANDMARK_ALIGN = 1

g_align_method = OPENCV_ALIGN

work_root = "D:\\Master\\Public\\deeplearning\\deeplearning\\"
cv_root = 'D:/Master/Public/opencv/'
caffe_root = 'D:/Work/Python/imgworks/'

#face alignment and crop 
exe_path = 'D:\\Master\\Public\\deeplearning\\deeplearning\\FaceAlignmentAndCrop\\ImageWorks.exe'
conf_path = 'D:\\Master\\Public\\deeplearning\\deeplearning\\data\\config.json'

face_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_eye.xml')

caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root+'vgg_face_caffe/VGG_FACE_deploy.prototxt', caffe_root+'vgg_face_caffe/VGG_FACE.caffemodel')

def OutDebugInfo(a,*b):
    #return
    sys.stdout.write(a)
    for x in b:
        sys.stdout.write(x)
        
    print('')
    return

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.actionOpen_File.triggered.connect(self.openfile)
        self.actionOpencv_Align.triggered.connect(self.opencv_align)
        self.actionFlandmark_Align.triggered.connect(self.flandmark_align)
        self.actionAbout.triggered.connect(self.about)
        self.actionStart.triggered.connect(self.start)
        self.actionStop.triggered.connect(self.stop)
        self.identification.clicked.connect(self.capture_and_idfy)
        
        self.startcamera = False
        self.curfilepath = None
        self.cropfilepath = None
        self.loadimage = False
        
    def about(self):
        QMessageBox.about(self, "About the system", self.tr("基于vggface\n人脸识别系统"))
        
    def opencv_align(self):
        OutDebugInfo("OPENCV_ALIGN")
        g_align_method = OPENCV_ALIGN
    
    def flandmark_align(self):
        OutDebugInfo("FLANDMARK_ALIGN")
        g_align_method = FLANDMARK_ALIGN  
        
    def capture_img(self):
        self.capture_diag.cp.capture()
        
    def capture_img_handler(self, requestId, img):
        scaledImage = img.scaled(self.capture_diag.view_finder.size(), 1, 1)
        qpm = QPixmap.fromImage(scaledImage)
        
        self.curfilepath = work_root+'data\\work\\cur_online.jpg'
        qpm.save(self.curfilepath)
        
        img = cv2.imread(self.curfilepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces is None or len(faces)==0:
            QMessageBox.warning(self, "Warning", self.tr("您载入的图片中没有人脸"))
            return
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        
        cv2.imwrite(work_root+"data/work/new_detected.jpg", img)
        
        OutDebugInfo("display")
        
        imgdrawed = QImage(work_root+"data/work/new_detected.jpg")
        qpm = QPixmap.fromImage(imgdrawed)
        miniqpm = qpm.scaled(341, 361, QtCore.Qt.KeepAspectRatio)
        self.camera.setPixmap(miniqpm)
        self.loadimage = True
                   
        OutDebugInfo("destroy")
        
        self.capture_diag.destroy()
        
    def save_img(self):
        pass    
        
    def start(self):
        OutDebugInfo("start")
        diag = QDialog(self)        
        diag.setWindowTitle("Camera")

        # main layout
        diag.lay = QtWidgets.QVBoxLayout()

        # widgets
        diag.capture_button = QtWidgets.QPushButton("Capture")
        diag.capture_button.clicked.connect(self.capture_img)
        diag.label = QtWidgets.QLabel("")

        # setting the device
        diag.device = QCamera.availableDevices()[0]
        diag.m_camera = QCamera(diag.device)

        diag.view_finder = QtMultimediaWidgets.QCameraViewfinder()
        diag.view_finder.setMinimumSize(250, 250)

        diag.m_camera.setViewfinder(diag.view_finder)
        diag.m_camera.setCaptureMode(QCamera.CaptureStillImage)
        
        try:
            diag.m_camera.start()
            self.startcamera = True
            
        except:
            pass

        diag.lay.addWidget(diag.label)
        diag.lay.addWidget(diag.view_finder)
        diag.lay.addWidget(diag.capture_button)
        diag.lay.addWidget(diag.label)
        diag.setLayout(diag.lay)
        
        diag.cp = QCameraImageCapture(diag.m_camera)
        diag.cp.imageCaptured.connect(self.capture_img_handler)
        diag.cp.imageSaved.connect(self.save_img)
        
        self.capture_diag = diag
        
        diag.show()
        diag.exec_()
    
    def stop(self):
        OutDebugInfo("stop")
        if self.startcamera:
            self.capture_diag.m_camera.stop()
            self.capture_diag.destroy()
            self.capture_diag = None
            self.startcamera = False
    
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

            self.curfilepath = work_root+'data\\work\\cur_online.jpg'
            qpm.save(self.curfilepath)
            self.loadimage = True
       
    def read_image(self,filename):  
        averageImg = [129.1863,104.7624,93.5940]  
        X=np.empty((1,3,224,224))  
        im1=skimage.io.imread(filename,as_grey=False)  
        image =skimage.transform.resize(im1,(224, 224))*255  
        X[0,0,:,:]=image[:,:,0]-averageImg[0]  
        X[0,1,:,:]=image[:,:,1]-averageImg[1]  
        X[0,2,:,:]=image[:,:,2]-averageImg[2]  
        
        return X
        
    def align_and_crop(self):
        OutDebugInfo("align_and_crop")
        OutDebugInfo(self.curfilepath)
        if g_align_method==FLANDMARK_ALIGN:
            os.system(exe_path+' '+conf_path+' '+self.curfilepath)
            self.cropfilepath = work_root+'data\\result\\cur_online.jpg'
        elif g_align_method==OPENCV_ALIGN:
            img = cv2.imread(self.curfilepath)
            if img is None:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                crop_img = img[y:(y+h), x:(x+w)]
                cv2.imwrite(work_root+'data\\result\\cur_online.jpg', crop_img)
            self.cropfilepath = work_root+'data\\result\\cur_online.jpg'
        
    def capture_and_idfy(self):
        OutDebugInfo("capture_and_idfy")
        if self.loadimage==False:
            QMessageBox.about(self, "Warning", self.tr("请先使用摄像头采集照片"))
            return
            
        self.align_and_crop()
        
        if  self.cropfilepath is None or self.cropfilepath=="":
            QMessageBox.about(self, "Warning", self.tr("请先使用摄像头采集照片"))
            return
        
        X = self.read_image(self.cropfilepath)
        shape_x = np.shape(X)[0]
        out = net.forward_all(blobs=['fc7'], data=X)
        feature = np.float64(out['fc7'])
        feature = np.reshape(feature,(shape_x, 4096))
        
        self.face_feature_comp(feature)
    
    def clear_top_info(self):
        self.top1.clear()
        self.top2.clear()
        self.top3.clear()
        self.top4.clear()
        self.top5.clear()
        
        self.top1_name.clear()
        self.top2_name.clear()
        self.top3_name.clear()
        self.top4_name.clear()
        self.top5_name.clear()
        
    def face_feature_comp(self, feature):
        dataset_path = work_root + 'data/fc7_feature_npy/'
        most_similar_degree = 0.0
        list_similar_name = []
        list_similar_path = []
        
        self.clear_top_info()
        
        for item in os.listdir(dataset_path):
            curfile = dataset_path+item
            if os.path.isfile(curfile):
                if os.path.splitext(curfile)[1] in ['.npy']:
                    curdata = np.load(curfile)
                    predicts = pw.cosine_similarity(feature, curdata)
                    similar_rate = np.add.reduce(predicts)[0]
                    list_similar_name.append(item.split('.')[0])
                    list_similar_path.append(similar_rate)
        
        sorted_in_sr_index = np.argsort(list_similar_path) # increase sort
        sorted_sr_index = sorted_in_sr_index[::-1] # decrease sort
        lens = len(list_similar_path)
        OutDebugInfo(str(list_similar_path[sorted_sr_index[0]]))
        OutDebugInfo(str(list_similar_path[sorted_sr_index[1]]))
        OutDebugInfo(str(list_similar_path[sorted_sr_index[2]]))
        OutDebugInfo(str(list_similar_path[sorted_sr_index[3]]))
        OutDebugInfo(str(list_similar_path[sorted_sr_index[4]]))
        numk = 0
        if list_similar_path[sorted_sr_index[0]] > 0.45:
            self.top1_name.setText('1 '+list_similar_name[sorted_sr_index[0]].split('_')[1]+' '+str(list_similar_path[sorted_sr_index[0]]))
            strimgpath = work_root+"data/face_aligned/"+list_similar_name[sorted_sr_index[0]]+".JPG"
            imgdrawed = QImage(strimgpath)
            qpm = QPixmap.fromImage(imgdrawed)
            miniqpm = qpm.scaled(91, 91, QtCore.Qt.KeepAspectRatio)
            self.top1.setPixmap(miniqpm)
            numk = numk + 1

            if lens>1 and list_similar_path[sorted_sr_index[1]] > 0.5:
                self.top2_name.setText('2 '+list_similar_name[sorted_sr_index[1]].split('_')[1]+' '+str(list_similar_path[sorted_sr_index[1]]))
                strimgpath = work_root+"data/face_aligned/"+list_similar_name[sorted_sr_index[1]]+".JPG"
                imgdrawed = QImage(strimgpath)
                qpm = QPixmap.fromImage(imgdrawed)
                miniqpm = qpm.scaled(91, 91, QtCore.Qt.KeepAspectRatio)
                self.top2.setPixmap(miniqpm)
                numk = numk + 1
                
            if lens>2 and list_similar_path[sorted_sr_index[2]] > 0.5:
                self.top3_name.setText('3 '+list_similar_name[sorted_sr_index[2]].split('_')[1]+' '+str(list_similar_path[sorted_sr_index[2]]))
                strimgpath = work_root+"data/face_aligned/"+list_similar_name[sorted_sr_index[2]]+".JPG"
                imgdrawed = QImage(strimgpath)
                qpm = QPixmap.fromImage(imgdrawed)
                miniqpm = qpm.scaled(91, 91, QtCore.Qt.KeepAspectRatio)
                self.top3.setPixmap(miniqpm)
                numk = numk + 1
                
            if lens>3 and list_similar_path[sorted_sr_index[3]] > 0.5:
                self.top4_name.setText('4 '+list_similar_name[sorted_sr_index[3]].split('_')[1]+' '+str(list_similar_path[sorted_sr_index[3]]))
                strimgpath = work_root+"data/face_aligned/"+list_similar_name[sorted_sr_index[3]]+".JPG"
                imgdrawed = QImage(strimgpath)
                qpm = QPixmap.fromImage(imgdrawed)
                miniqpm = qpm.scaled(91, 91, QtCore.Qt.KeepAspectRatio)
                self.top4.setPixmap(miniqpm)
                numk = numk + 1
                
            if lens>4 and list_similar_path[sorted_sr_index[4]] > 0.5:
                self.top5_name.setText('5 '+list_similar_name[sorted_sr_index[4]].split('_')[1]+' '+str(list_similar_path[sorted_sr_index[4]]))
                strimgpath = work_root+"data/face_aligned/"+list_similar_name[sorted_sr_index[4]]+".JPG"
                imgdrawed = QImage(strimgpath)
                qpm = QPixmap.fromImage(imgdrawed)
                miniqpm = qpm.scaled(91, 91, QtCore.Qt.KeepAspectRatio)
                self.top5.setPixmap(miniqpm)
                numk = numk + 1
                
            self.top_k_label.setText('Top-'+str(numk)+' Results')
        else:
            QMessageBox.about(self, "Warning", self.tr("数据库中无与此人脸接近的数据，识别失败！"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    
    sys.exit(app.exec_())