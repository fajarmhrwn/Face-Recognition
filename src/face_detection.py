import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from eigenface import *

def convertImage(imagename):
    # Mengubah image menjadi matriks n*n x 1
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

def getMean(pth):
    # Mencari rata rata matriks dari sekumpulan foto yang diubah ke matriks
    mean = [0 for i in range(256*256)]
    c=0
    path = r"test/pins_dataset/" + pth #"test/pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    for file in temp:
        a = convertImage("test/pins_dataset/"+pth+"/"+file)
        c+=1
        for i in range (256*256):
            mean[i] += a[i]
    for i in range (256*256):
        mean[i] /= c
    return mean

def getCovariance(mean,pth):
    # Mencari nilai matriks kovarian dari input matriks mean
    mean = np.reshape(mean,(256,256))
    c=0
    path = r"test/pins_dataset/" + pth #"test/pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    cov = np.zeros((256, 256))
    selisih = np.zeros((256, 256))
    for file in temp:
        a = convertImage("test/pins_dataset/"+pth+"/"+file)
        a = np.reshape(a,(256,256))
        c+=1
        selisih = np.subtract(a,mean)
        selisihTranspose = np.transpose(selisih)
        # Sesuai persamaan covariant
        cov = np.add(cov,np.matmul(selisih,selisihTranspose))
    cov = cov/c
    return cov

def getBanyakFoto(pth):
    # Mencari banyaknya file foto pada suatu folder
    path = r"test/pins_dataset/" + pth #"pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    c = 0
    for file in temp:
        c += 1
    return c



in_folder_name = input("Masukkan nama folder dataset: ")
NFoto = getBanyakFoto(in_folder_name)
a = getMean(in_folder_name) 
b = getCovariance(a,in_folder_name) 
# print(b)
c = findeigenface(b,in_folder_name)
# print(c)
plt.imshow(c,cmap='gray')
plt.show()


    

# out_file = open("test\meanface\\"+  in_folder_name +".txt", "w+")
# content = str(a)
# out_file.write(content)
# out_file.close()