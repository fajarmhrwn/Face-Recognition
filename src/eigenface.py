# Algoritma eigenface , ngitung eigenvalue pake QR algorithm
import numpy as np
import os
import cv2

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

def list_eigenface(path):
    # Mencari eigenface untuk semua dataset
    print("Mencari eigenface")
    temp = f"test/pins_dataset/{path}"
    allImage = np.empty((256*256,0), float)
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                convertedImage = convertImage(tempPath)
                # print(convertedImage.shape)
                allImage= np.column_stack((allImage, convertedImage.reshape(256*256, 1)))
    mean_subtracted = allImage - allImage.mean(axis=1, keepdims=True)
    redCov = np.matmul(np.transpose(mean_subtracted),mean_subtracted)
    eigenvalue, eigenvector = eigen_qr_practical(redCov)
    # multiply with eigen vector
    grthnOne = 0
    for i in eigenvalue.diagonal():
        if i > 1:
            grthnOne += 1
    redEigenVector = eigenvector[:, :grthnOne]
    bestEigenVectorsOfCov = np.empty((256*256, 0), float)
    for i in range(len(redEigenVector[0])) :
        temp = np.matmul(mean_subtracted, np.transpose([redEigenVector[:, i]]))
        bestEigenVectorsOfCov = np.column_stack((bestEigenVectorsOfCov, temp))
    
    return eigenvector[:,:grthnOne],bestEigenVectorsOfCov


def cropAllImage(path):
    # Mengcrop semua image
    print("Mengcrop semua image")
    temp = f"test/pins_dataset/{path}"
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                cropimage(tempPath)
    print("Selesai crop semua image")

cropAllImage("test2")