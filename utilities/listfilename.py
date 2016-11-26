# -*- coding: utf-8 -*-

import os

work_path = "D:\\Master\\Public\\deeplearning\\deeplearning\\"

if __name__ == '__main__':
    targetdir = work_path + "data\\datasets"
    f = open("filename.txt","w")
    for item in os.listdir(targetdir):
        print(item)
        f.writelines(item+"\n")

    f.close()
        
