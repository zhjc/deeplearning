# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:03:56 2016

@author: chinn
"""

import numpy
import cv2
import os

cv_root = 'D:/Master/Public/opencv/'
work_root = 'D:/Work/Python/imgworks/'

file_ext = ['.JPG','.jpg','.PNG','.png', '.JPEG', '.jpeg']

face_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv_root+'data/haarcascades/haarcascade_eye.xml')

def crop_images(filepath, filename):
    img = cv2.imread(filepath)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = img[y:(y+h), x:(x+w)]
        cv2.imwrite(work_root+"face_icon/"+filename+'.jpg', crop_img)
        
if __name__ == '__main__':
    targetdir = work_root+'datasets/'
    for item in os.listdir(targetdir):
        curfile = targetdir+item
        if os.path.isfile(curfile):
            if os.path.splitext(curfile)[1] in file_ext:
                crop_images(curfile, item.split('.')[0])