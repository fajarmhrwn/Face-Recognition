# pake opencv buat ngambil wajah dr kamera
# dilanjut sampe konversi greyscale, kebentuk matriks 

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def convertImage(imagename):
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

def getMean(pth):
    mean = [0 for i in range(256*256)]
    c=0
    path = r"pins_dataset/" + pth #"pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    for file in temp:
        a = convertImage(pth+"/"+file)
        c+=1
        for i in range (256*256):
            mean[i] += a[i]
    for i in range (256*256):
        mean[i] /= c
    return mean

a = getMean("pins_Jimmy Fallon")
reshape = np.reshape(a,(256,256))
plt.imshow(reshape, cmap='gray')
plt.show()
