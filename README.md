# Deeplearning Assignment 

<font size=4>face identification based vggface caffe pre-trained model</font>

## 基于vggface的人脸识别系统

### 系统框架
![](docs/face_identification_framework.png)

### 系统使用说明

运行系统需要opencv和PyQt5支持，推荐使用Anaconda集成环境。

```cmd
run.bat
```

在运行run.bat文件之前，需要增加一些配置，修改src/faceidentification.py文件中一些路径
```cmd
work_root = "D:\\Master\\Public\\deeplearning\\deeplearning\\"
cv_root = 'D:/Master/Public/opencv/'
caffe_root = 'D:/Work/Python/imgworks/'
exe_path = 'D:\\Master\\Public\\deeplearning\\deeplearning\\FaceAlignmentAndCrop\\ImageWorks.exe'
conf_path = 'D:\\Master\\Public\\deeplearning\\deeplearning\\config.json'
```

