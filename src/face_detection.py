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
        a = a.reshape(256*256, 1)
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



def getLinComOfEigVector(bestEigenVectorsOfCov, imageVectorInput) :
    """
    return the linear Combination of bestEigenVectorsOfCov from imageVectorInput
    size : number of best eigenfaces x 1
    """
    x = bestEigenVectorsOfCov
    y = np.transpose(imageVectorInput)
    linCom = np.transpose([np.linalg.lstsq(x, y[0], rcond=None)[0]])
    return linCom


def getMinimumDistance(inputLinCom, CoefMatrix) :
    """
    return minimum distance from linear combination of input image 
    and linear combination of each image in data set
    """
    minimum = minimum = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, 0]])))
    for i in range(len(CoefMatrix[0])) :
        distance = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, i]])))
        if (distance < minimum) :
            minimum = distance
    return minimum


def getNearestImage(dataset_folder,file_path, eigenvector, eigenface):
    path = f"test/input/{file_path}"#"pins_dataset" itu nama foldernya
    tes_image = np.empty((256*256,0), float) 
    a = convertImage(path)
    tes_image = np.column_stack((tes_image,a.reshape(256*256,1)))
    temp = f"test/pins_dataset/{dataset_folder}"
    allImage = np.empty((256*256,0), float)
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                image = cv2.imread(tempPath, 0) # foto grayscale yang udah 256x256
                convertedImage = convertImage(tempPath)
                # print(convertedImage.shape)
                print(convertedImage.reshape(256*256,1))
                allImage= np.column_stack((allImage, convertedImage.reshape(256*256, 1)))
    
    mean = allImage.mean(axis=1, keepdims=True)
    print(mean.shape)
    print(mean)
    # hasil_kurang = tes_image - mean
    # print(hasil_kali.shape)
    hasil_kali_test = np.empty((256*256,0), float) 
    for i in range(len(eigenvector[0])) :
        hasil_kali = np.matmul(hasil_kurang,np.transpose(eigenvector[:,i]))
        hasil_kali_test = np.column_stack(hasil_kali_test, hasil_kali)
    str_name_nearest = ""
    maxNorm = 0
    temp = os.listdir(dataset_folder)
    for file in temp:
        if(getNorm(a-eigenface[:,i]) > maxNorm):
            maxNorm = getNorm(a-eigenface[:,i])
            str_name_nearest = file
        i += 1
    return str_name_nearest


# Start
in_folder_name = input("Masukkan nama folder dataset: ")
    

eigenvector, eigenface = list_eigenface(in_folder_name)

print(eigenface,eigenface.shape)
#show one of image
#reshape to 256x256
for i in range(10):
    img = np.reshape(eigenface[:,i+1],(256,256))
    img *= 255/img.max()
    plt.imshow(img, cmap='gray')
    #save image
    cv2.imwrite("hasil"+ str(i) + ".jpg",img)


file_input = input("Masukkan nama file foto input di folder test/input : ")

dataset_folder = r"test/pins_dataset/" + in_folder_name #"test/pins_dataset" itu nama foldernya

closest_image = getNearestImage(dataset_folder,file_input,eigenvector, eigenface)
print(closest_image)

# out_file = open("test\meanface\\"+  in_folder_name +".txt", "w+")
# content = str(a)
# out_file.write(content)
# out_file.close()