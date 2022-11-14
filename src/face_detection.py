import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import eigenface as eigenface

def convertImage(imagename):
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

def getMean(pth):
    mean = [0 for i in range(256*256)]
    c=0
    path = r"test/pins_dataset/" + pth #"pins_dataset" itu nama foldernya
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
    mean = np.reshape(mean(256,256))
    c=0
    path = r"test/pins_dataset/" + pth #"pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    cov = np.zeros((256, 256))
    selisih = np.zeros((256, 256))
    for file in temp:
        a = convertImage("test/pins_dataset/"+pth+"/"+file)
        a = np.reshape(a,(256,256))
        c+=1
        # for i in range (256):
        #     for j in range (256):
        #         selisih[i][j] += (a[i][j]-mean[i][j])
        selisih = np.subtract(a,mean)
        selisihTranspose = np.transpose(selisih)
        cov = np.add(cov,np.matmul(selisih,selisihTranspose))
    cov = cov/c
    return cov

def getBanyakFoto(pth):
    path = r"test/pins_dataset/" + pth #"pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    c = 0
    for file in temp:
        c += 1
    return c

in_folder_name = input("Masukkan nama folder dataset: ")
NFoto = getBanyakFoto(in_folder_name)
a = getMean(in_folder_name) #ini ambil contoh ajaya ges
b = getCovariance(a,in_folder_name) 
c = eigenface.eigenface(b,NFoto)


    

# out_file = open("test\meanface\\"+  in_folder_name +".txt", "w+")
# content = str(a)
# out_file.write(content)
# out_file.close()

# reshape = np.reshape(a,(256,256))
# plt.imshow(reshape, cmap='gray')
# plt.show()

