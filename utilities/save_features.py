# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27

@author: chinn
"""

import caffe
import numpy as np
import os
import skimage
import sklearn.metrics.pairwise as pw

caffe_root = 'D:/Work/Python/imgworks/'
file_ext = ['.JPG','.jpg','.PNG','.png', '.JPEG', '.jpeg']

import sys  
sys.path.insert(0, caffe_root + 'python')

def read_image(filename):  
    averageImg = [129.1863,104.7624,93.5940]  
    X=np.empty((1,3,224,224))  
    im1=skimage.io.imread(filename,as_grey=False)  
    image =skimage.transform.resize(im1,(224, 224))*255  
    X[0,0,:,:]=image[:,:,0]-averageImg[0]  
    X[0,1,:,:]=image[:,:,1]-averageImg[1]  
    X[0,2,:,:]=image[:,:,2]-averageImg[2]  
    
    return X
    
def compare_pic(img1fullpath, filename, net):
    X = read_image(img1fullpath)
    shape_x = np.shape(X)[0]
    out1 = net.forward_all(blobs=['fc7'], data=X)
    feature1 = np.float64(out1['fc7'])
    feature1 = np.reshape(feature1,(shape_x, 4096))
    np.save(caffe_root+'face_icon_feature/'+filename+'.npy', feature1)
    #data = np.load(caffe_root+'fc7_feature_bin/'+filename+'.npy')
    
if __name__ == '__main__':
    caffe.set_mode_cpu()
    net = caffe.Classifier(caffe_root+'vgg_face_caffe/VGG_FACE_deploy.prototxt', caffe_root+'vgg_face_caffe/VGG_FACE.caffemodel')
    
    targetdir = caffe_root+'face_icon/'
    for item in os.listdir(targetdir):
        curfile = targetdir+item
        if os.path.isfile(curfile):
            if os.path.splitext(curfile)[1] in file_ext:
                compare_pic(curfile, os.path.splitext(item)[0], net)
                #break
   
    