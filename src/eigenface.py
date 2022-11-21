# Algoritma eigenface , ngitung eigenvalue pake QR algorithm
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

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
    # Algoritma QR untuk mencari eigenvalue dan eigenvector
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
    return Ak,QQ # QQ adalah eigenvektor


def convertImage(imagename):
    # Mengubah image menjadi matriks n*n x 1
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

#ini buat ngambil nilai linear kombinasi dari setiap gamabar di dataset
#normalizedDataset = mean subtracted
def getLinComMatrix(bestEigenVector, normalizedDataSet) :
    CoefOfLinComMatrix = np.empty((len(bestEigenVector[0]),0), float)

    for i in range(len(normalizedDataSet[0])) :
        LinComOfNormalized = getLinComOfEigVector(bestEigenVector, np.transpose([normalizedDataSet[:,i]]))
        CoefOfLinComMatrix = np.column_stack((CoefOfLinComMatrix, LinComOfNormalized))
    
    return CoefOfLinComMatrix

#ini buat ngambil nilai linear kombinasi dari setiap gamabar di dataset nanti disimpan disuatu file

#Linear kombinasi dari satu gambar
def getLinComOfEigVector(bestEigenVectorsOfCov, imageVectorInput) :
    x = bestEigenVectorsOfCov
    # x = (256*256, 1)
    y = np.transpose(imageVectorInput)# ini matriks 1 , 256*256
    linCom = np.transpose([np.linalg.lstsq(x, y[0], rcond=None)[0]])
    return linCom

#inputannya tadi linear kombinasi gambar input dan matrix tadi 
def getMinimumDistance(inputLinCom, CoefMatrix) :
    minimum = minimum = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, 0]])))
    for i in range(len(CoefMatrix[0])) :
        distance = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, i]])))
        if (distance < minimum) :
            minimum = distance
    return minimum


def cropimage(image):
    # Mengcrop image agar hanya mengambil bagian wajah
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('src/face.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h+10, x:x+w+10]
            cv2.imwrite(image, crop_img)
            return image
    else:
        print("Tidak ada")
        os.remove(image)
        return image

def list_eigenface(mean_subtracted,all_image):
    # Mencari eigenface untuk semua dataset
    print("Mencari eigenface")
    print(mean_subtracted.shape)#(256*256, 100)
    redCov = np.matmul(np.transpose(mean_subtracted),mean_subtracted) # A'A  = (122, 122)
    #nilai bes eigenvectornya disimpan di file
    eigenvalue, eigenvector = eigen_qr_practical(redCov)
    print(eigenvector.shape)
    # multiply with eigen vector
    grthnOne = 0
    for i in eigenvalue.diagonal():
        if i > 1:
            grthnOne += 1
    redEigenVector = eigenvector[:, :grthnOne] #(122,121)
    print(redEigenVector[:,0].shape)
    print(redEigenVector[:,0])
    print(np.transpose([redEigenVector[:,0]]).shape)
    print(np.transpose(redEigenVector[:,0]))
    bestEigenVectorsOfCov = np.empty((256*256, 0), float)
    for i in range(len(redEigenVector[0])) :
        temp = np.matmul(all_image, np.transpose([redEigenVector[:, i]]))
        bestEigenVectorsOfCov = np.column_stack((bestEigenVectorsOfCov, temp))
    
    print(bestEigenVectorsOfCov.shape)
    return bestEigenVectorsOfCov
    #ini disave

def cropAllImage(path):
    # Mengcrop semua image
    print("Mengcrop semua image")
    temp = f"test/pins_dataset/{path}"
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                cropimage(tempPath)
    print("Selesai crop semua image")



def getClosestImage (dirPath, CoefMatrix, inputLinCom) :
    minimum = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, 0]])))
    imageOrder = 1
    for i in range(len(CoefMatrix[0])) :
        distance = np.linalg.norm(np.subtract(inputLinCom, np.transpose([CoefMatrix[:, i]])))
        if (distance < minimum) :
            minimum = distance
            imageOrder = i + 1

    count = 0
    for (dirPath, dirNames, file) in os.walk(dirPath):
        for fileNames in file :
            count += 1
            if count == imageOrder :
                return os.path.join(dirPath, fileNames)


def gettrainingdataa(path):
    temp = f"test/pins_dataset/{path}"
    allImage = np.empty((256*256,0), float)
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                convertedImage = convertImage(tempPath)
                # print(convertedImage.shape)
                allImage= np.column_stack((allImage, convertedImage.reshape(256*256, 1)))
    print(allImage.shape)
    mean_subtracted = allImage - allImage.mean(axis=1, keepdims=True)
    eigeface = list_eigenface(mean_subtracted,allImage)
    CoefMatrix = getLinComMatrix(eigeface, mean_subtracted)
    #save coefmatrix dan eigenface
    # np.savetxt(f"data/coefmatrix.txt", CoefMatrix, delimiter=",")
    # np.savetxt(f"data/eigenface.txt", eigeface, delimiter=",")
    return CoefMatrix, eigeface


def generateclosestimage(path, lincom_imageinput, lincomdataset):
    # Mencari gambar terdekat
    print("Mencari gambar terdekat")
    temp = f"test/pins_dataset/{path}"
    minimum = getMinimumDistance(lincom_imageinput, lincomdataset)
    print(minimum)
    if minimum <  5 :    
        print("Gambar terdekat")
        closestimagefile =  getClosestImage(temp, lincomdataset, lincom_imageinput)
        print(closestimagefile)
        return closestimagefile
    else:
        print(minimum)
        print("Gambar tidak ditemukan")
        return None



def displayEigenFace(eigenFace) :
    # Menampilkan eigenface
    print("Menampilkan eigenface")
    fig = plt.figure()
    for i in range(len(eigenFace[0])) :
        fig.add_subplot(10, 16, i+1)
        plt.imshow(eigenFace[:, i].reshape(256, 256), cmap='gray')
    plt.show()
# cropimage("test/input/test_fajar.png")
print("Mulai")
path = input("Masukkan nama folder dataset: ")
coefmatrix, eigenface = gettrainingdataa(path)
displayEigenFace(eigenface)
# image_path = input("Masukkan nama file gambar: ")
# Image = convertImage("test/input/"+image_path)
# Image = Image.reshape(256*256, 1)
# lincom_imageinput = getLinComOfEigVector(eigenface, Image)
# path = generateclosestimage(path, lincom_imageinput, coefmatrix)
# #show image
# if path != None:
#     img = cv2.imread(path)
#     cv2.imwrite("test/input/1"+image_path, img)
