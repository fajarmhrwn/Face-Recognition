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

def getEigenFace(V,path,mean):
    # Mencari eigenface dari matriks eigenvektor
    temp = os.listdir(path)
    EF = np.zeros((len(V),len(V)))
    for file in temp:
        a = convertImage(path+"/"+file)
        a = np.reshape(a,(256,256))
        mean = np.reshape(mean,(256,256))
        a = a-mean
        EF += np.matmul(a,V)
    return EF


def convertImage(imagename):
    # Mengubah image menjadi matriks n*n x 1
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted

def findeigenface(A,pth,mean):
    # Mencari eigenface untuk satu dataset  
    path = r"test/pins_dataset/" + pth #"test/pins_dataset" itu nama foldernya
    eigenValueMat,eigenVectorMat = eigen_qr_practical(A)
    print(eigenValueMat,eigenVectorMat)
    print(np.diagonal(eigenValueMat))
    eigenfaceMat = getEigenFace(eigenVectorMat,path,mean)
    return eigenfaceMat


def getEignValuesVectors(Matrix):
    """ 
    return all eign value and vectors of A by 
    Simultaneous power iteration method 
    """
    rowNum = Matrix.shape[0] 

    # Initialize the eigenvectors
    # *** QR decomposition ***
    Q = np.random.rand(rowNum, rowNum) 
    Q, _ = np.linalg.qr(Q)
    Q_prev = np.zeros((rowNum, rowNum)) # Initialize previous Q

    # Initialize the eigenvalues
    eVal = np.zeros(rowNum) 
    
    # Iterate until convergence
    while np.linalg.norm(Q - Q_prev) > 1e-10: # Convergence criterion
        # *** Update previous Q ***
        Q_prev = Q 

        # Compute the matrix-by-vector product AZ
        Z = Matrix.dot(Q) 
        # Compute the QR factorization of Z
        Q, _ = np.linalg.qr(Z) 

        # Update the eigenvalues
        eVal = np.diag(Q.T.dot(Matrix.dot(Q)))
    return eVal, Q


def list_eigenface(path):
    # Mencari eigenface untuk semua dataset
    print("Mencari eigenface")
    temp = f"test/pins_dataset/{path}"
    allImage = np.empty((256*256,0), float)
    for (dirPath, dirNames, file) in os.walk(temp):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                image = cv2.imread(tempPath, 0) # foto grayscale yang udah 256x256
                convertedImage = convertImage(tempPath)
                # print(convertedImage.shape)
                allImage= np.column_stack((allImage, convertedImage.reshape(256*256, 1)))
    mean_subtracted = allImage - allImage.mean(axis=1, keepdims=True)
    redCov = np.matmul(np.transpose(mean_subtracted),mean_subtracted)
    eigenvalue, eigenvector = eigen_qr_practical(redCov)
    '''Eliminating some eigenvectors'''
    grthnOne = 0
    for i in eigenvalue.diagonal():
        if i > 1:
            grthnOne += 1
    '''Take the best eigenvalue bigger than 1'''
    redEigenVector = eigenvector[:, :grthnOne]
    bestEigenVectorsOfCov = np.empty((256*256, 0), float)
    for i in range(len(redEigenVector[0])) :
        temp = np.matmul(mean_subtracted, np.transpose([redEigenVector[:, i]]))
        print("cek")
        print(np.shape(mean_subtracted), np.shape(redEigenVector))
        bestEigenVectorsOfCov = np.column_stack((bestEigenVectorsOfCov, temp))
    
    return eigenvector[:,:grthnOne],bestEigenVectorsOfCov


