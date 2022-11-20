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
    
    return greyscaleimg

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

def getCovariance(mean,pth,Nfoto):
    # Mencari nilai matriks kovarian dari input matriks mean
    # mean = np.reshape(mean,(256,256))
    # mean = np.transpose(mean)
    c=0
    path = r"test/pins_dataset/" + pth #"test/pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    cov = np.empty([0,256* 256])
    for file in temp:
        a = convertImage("test/pins_dataset/"+pth+"/"+file)
        # a = np.transpose(a)
        # a = np.reshape(a,(256,256))
        c+=1
        selisih = np.subtract(a,mean)
        selisih = np.reshape(selisih,[1,256*256])
        # print(np.shape(cov), np.shape(selisih))
        cov = np.append(cov,selisih, axis=0)
        # print(cov,"fsdffsdfsf")
        # selisihTranspose = np.transpose(selisih)
        # Sesuai persamaan covariant
        # cov = np.add(cov,np.matmul(selisih,selisihTranspose))
    
    cov = np.matmul(cov,np.transpose(cov))
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


eigenface = list_eigenface(in_folder_name)

print(eigenface,eigenface.shape)
#show one of image
#reshape to 256x256
img = np.reshape(eigenface[:,4],(256,256))
img *= 255/img.max()
plt.imshow(img, cmap='gray')
#save image
cv2.imwrite("hasil.jpg",img)




    

# out_file = open("test\meanface\\"+  in_folder_name +".txt", "w+")
# content = str(a)
# out_file.write(content)
# out_file.close()