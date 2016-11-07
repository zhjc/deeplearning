# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26

@author: chinn
"""

import sys, os

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize 
from ui_mainwindow import Ui_ImageWorks

from PyQt5 import QtWidgets, QtMultimediaWidgets, QtCore
from PyQt5.QtMultimedia import QCamera,QCameraImageCapture

import caffe
import cv2
import skimage
import numpy as np

import sklearn.metrics.pairwise as pw

work_root = 'D:/Work/Python/imgworks/'
sys.path.insert(0, work_root + 'work')

cv_root = 'D:/Master/Public/opencv/'

face_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_eye.xml')

caffe.set_mode_cpu()
net = caffe.Classifier(work_root+'vgg_face_caffe/VGG_FACE_deploy.prototxt', work_root+'vgg_face_caffe/VGG_FACE.caffemodel')
vgg16net = None#caffe.Net(work_root+'vgg16/VGG_ILSVRC_16_layers_deploy.prototxt', work_root+'vgg16/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
 
class MyWindow(QMainWindow,Ui_ImageWorks):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.actionAbout.triggered.connect(self.about)
        self.actionOpen_File.triggered.connect(self.openfile) #上传文件
        self.pushButton_2.clicked.connect(self.ic_)
        self.pushButton_3.clicked.connect(self.fd_)
        self.pushButton_4.clicked.connect(self.fi_)
        self.pushButton_5.clicked.connect(self.fio_)
        
        self.label_qpiname = ''
        self.label_2_qpiname = ''
        self.capture_diag = None
        self.onlinefi = False
        self.onlinefi_img_path = ''
        
    def about(self):
        QMessageBox.about(self, "About", self.tr("图形识别系统"))
 
    def openfile(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Select File", "C:/", "jpg Files (*.jpg);;png Files (*.png)")
        if filename is None:
            return
        img = QImage(filename)
        if img is None:
            QMessageBox.warning(self, "Warning", self.tr("图片载入失败，请重新操作"))
            return
        else:
            qpm = QPixmap.fromImage(img)
            miniqpm = qpm.scaled(290,360,QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(miniqpm)
            self.label_qpiname = filename
            self.onlinefi = False
    
    def capture_img(self):
        self.capture_diag.cp.capture()
        
    def capture_img_handler(self, requestId, img):
        scaledImage = img.scaled(self.capture_diag.view_finder.size(), 1, 1)
        qpm = QPixmap.fromImage(scaledImage)
        
        self.onlinefi_img_path = work_root+'work/cur_online.jpg'
        qpm.save(self.onlinefi_img_path)
        miniqpm = qpm.scaled(290,360,QtCore.Qt.KeepAspectRatio)
        self.label_2.setPixmap(miniqpm)
        #self.label_2.resize(QSize(scaledImage.width(),scaledImage.height()))
        #self.label_2.resize(scaledImage.width(),scaledImage.height())
        
        self.capture_diag.m_camera.stop()
        self.capture_diag.destroy()
        self.capture_diag = None
        self.onlinefi = True
        
        self.face_identification_online(self.onlinefi_img_path)
        
    def save_img(self):
        pass    
        
    # 人脸检测槽函数
    def fd_(self):
        qpi = self.label_2.pixmap()
        if qpi is not None:
            self.face_detected(self.label_qpiname)
            #debug
            #print(self.label_qpiname)
        else:
            QMessageBox.warning(self, "Warning", self.tr("人脸检测之前请先载入图片"))
            return
    
    # 物体分类槽函数
    def ic_(self):
        qpi = self.label_2.pixmap()
        if qpi is not None:
            self.img_classify(self.label_qpiname) 
        else:
            QMessageBox.warning(self, "Warning", self.tr("物体分类之前请先载入图片"))
            return
    
    # 本地人脸识别槽函数，直接使用图片进行识别
    def fi_(self):
        qpi = self.label_2.pixmap()
        if qpi is not None:
            self.face_identification(self.label_qpiname) 
        else:
            QMessageBox.warning(self, "Warning", self.tr("人脸识别之前请先载入图片"))
            return
    
    # 在线人脸识别槽函数，直接调用摄像头
    def fio_(self):
        diag = QDialog(self)        
        diag.setWindowTitle("Capture")

        # main layout
        diag.lay = QtWidgets.QVBoxLayout()

        # widgets
        diag.capture_button = QtWidgets.QPushButton("Capture")
        diag.capture_button.clicked.connect(self.capture_img)
        diag.label = QtWidgets.QLabel("")

        # setting the device
        diag.device = QCamera.availableDevices()[0]
        diag.m_camera = QCamera(diag.device)

        #test
        #diag.label.setText(str(diag.m_camera.availableDevices()))

        diag.view_finder = QtMultimediaWidgets.QCameraViewfinder()
        diag.view_finder.setMinimumSize(250, 250)

        diag.m_camera.setViewfinder(diag.view_finder)
        diag.m_camera.setCaptureMode(QCamera.CaptureStillImage)
        
        try:
            diag.m_camera.start()
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
        
    def face_detected(self, filename):
        img = None
        if self.onlinefi:
            img = cv2.imread(self.onlinefi_img_path)
        else:
            img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces is None or len(faces)==0:
            QMessageBox.warning(self, "Warning", self.tr("您载入的图片中没有人脸"))
            return
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            
        #-----------------------------------------
        # up half of the face is set to find eyes!
        #-----------------------------------------
        roi_gray = gray[y:y+h/2, x:x+w]
        roi_color = img[y:y+h/2, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        #cv2.imshow('detected_img',img)
        cv2.imwrite(cv_root+"new_detected.jpg", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        self.label.setPixmap(QPixmap.fromImage(QImage(cv_root+"new_detected.jpg")))
        
    def img_classify(self, filename):
        from imagenet_classes import class_names

        transformer = caffe.io.Transformer({'data': vgg16net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        # 图片像素放大到[0-255]
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        vgg16net.blobs['data'].reshape(1, 3, 224, 224)
        
        im = None
        if self.onlinefi:
            im = caffe.io.load_image(self.onlinefi_img_path)
        else:
            im = caffe.io.load_image(filename)
        
        vgg16net.blobs['data'].data[...] = transformer.preprocess('data', im)
        
        #输出每层网络的name和shape
        #for layer_name, blob in vgg16net.blobs.iteritems():
        #    print layer_name + '\t' + str(blob.data.shape)
        
        output = vgg16net.forward()
        
        # 找出最大的那个概率
        output_prob = output['prob'][0]
        #print output_prob result:[0.001,0.001,...,0.001]
        
        self.label.clear()
        self.label.setText('img type index is: '+str(output_prob.argmax())+'\n\n'+class_names[output_prob.argmax()])
        
        # 找出最可能的前俩名的类别和概率
        #top_inds = output_prob.argsort()[::-1][:2]
        #print "possible two thing: ",top_inds
        #self.label.setText("most possible two things: ", output_prob[top_inds[0]], output_prob[top_inds[1]])
    
    def read_image(self, filename):  
        averageImg = [129.1863,104.7624,93.5940]  
        X=np.empty((1,3,224,224))  
        im1=skimage.io.imread(filename,as_grey=False)  
        image =skimage.transform.resize(im1,(224, 224))*255  
        X[0,0,:,:]=image[:,:,0]-averageImg[0]  
        X[0,1,:,:]=image[:,:,1]-averageImg[1]  
        X[0,2,:,:]=image[:,:,2]-averageImg[2]  
        
        return X
        
    def compare_pic(self, img1, img2):
        X = self.read_image(img1)
        shape_x = np.shape(X)[0]
        out1 = net.forward_all(blobs=['fc7'], data=X)
        feature1 = np.float64(out1['fc7'])
        feature1 = np.reshape(feature1,(shape_x, 4096))
        
        Y = self.read_image(img2)
        out2 = net.forward_all(blobs=['fc7'], data=Y)
        feature2 = np.float64(out2['fc7'])
        feature2 = np.reshape(feature2,(shape_x, 4096))
        
        predicts = pw.cosine_similarity(feature1, feature2)
        
        return predicts
    
    def face_feature_comp(self, feature):
        dataset_path = work_root + 'face_icon_feature/'
        most_similar_degree = 0.0
        most_similar_name = None
        most_similar_path = None
        
        for item in os.listdir(dataset_path):
            curfile = dataset_path+item
            if os.path.isfile(curfile):
                if os.path.splitext(curfile)[1] in ['.npy']:
                    curdata = np.load(curfile)
                    predicts = pw.cosine_similarity(feature, curdata)
                    similar_rate = np.add.reduce(predicts)[0]
                    print(curfile, similar_rate)
                    if similar_rate>most_similar_degree:
                        most_similar_degree = similar_rate
                        most_similar_name = item.split('.')[0]
                        most_similar_path = curfile
                        if most_similar_degree>0.8:
                            break
        
        self.label.clear()
        if most_similar_degree > 0.5:
            self.label.setText(self.tr("置信度：")+str(most_similar_degree)+ "\n" + self.tr("与图片中最接近的人脸为：")+most_similar_name)
            #self.label.setPixmap(QPixmap.fromImage(most_similar_path))
        else:
            self.label.setText(self.tr("数据库中无与此人脸接近的数据，识别失败！"))
            
    def crop_images(self, filepath, filename):
        img = cv2.imread(filepath)
        if img is None:
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            crop_img = img[y:(y+h), x:(x+w)]
            cv2.imwrite(work_root+"work/"+filename+'.jpg', crop_img)
        
    def face_identification(self, filename):
        self.crop_images(filename, "cur_idimpg")
        
        X = self.read_image(work_root+"work/cur_idimpg.jpg")
        
        shape_x = np.shape(X)[0]
        out = net.forward_all(blobs=['fc7'], data=X)
        feature = np.float64(out['fc7'])
        feature = np.reshape(feature,(shape_x, 4096))
        
        self.face_feature_comp(feature)
    
    def face_identification_online(self, filename):        
        self.face_identification(filename)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    
    sys.exit(app.exec_())
    