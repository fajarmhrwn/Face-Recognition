# Algoritma eigenface , ngitung eigenvalue pake QR algorithm
import numpy as np
import os
import cv2
from tabulate import tabulate

def getNorm(m):
    #Mendapatkan norm dari matriks
    matrix = np.square(m)
    matrix = np.sum(matrix)
    matrix = np.sqrt(matrix)
    
    return matrix

def checkConverge(arr,arr_prev):
    # Mengecek nilai elemen matriks apakah konvergen menuju suatu nilai
    # print(np.linalg.norm(arr_prev-arr))
    for i in range(len(arr)):
        for j in range(len(arr)):
            if (getNorm(arr_prev-arr)> 10000): # Parameter dapat diubah sesuai tingkat akurasi
                return False
    return True

def eigen_qr_practical(A):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    i = 0
    while(True):
        # s_k is the last item of the first diagonal
        Ak_copy = np.copy(Ak)
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        # pe perform qr and subtract smult
        Q, R = np.linalg.qr(np.subtract(Ak, smult))
        # we add smult back in
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q
        i += 1
        if(checkConverge(Ak,Ak_copy)):
            # print(QQ)
            break
    return QQ # QQ adalah eigenvektor

def getEigenFace(V,path):
    temp = os.listdir(path)
    EF = np.zeros((len(V),len(V)))
    for file in temp:
        a = convertImage(path+"/"+file)
        a = np.reshape(a,(256,256))
        EF += np.matmul(V,a)
    return EF


def convertImage(imagename):
    # Mengubah image menjadi matriks n*n x 1
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

def findeigenface(A,pth):
    path = r"test/pins_dataset/" + pth #"test/pins_dataset" itu nama foldernya
    eigenVectorMat = eigen_qr_practical(A)
    eigenfaceMat = getEigenFace(eigenVectorMat,path)
    return eigenfaceMat