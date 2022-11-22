import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

def getNorm(m):
    '''Mendapatkan norm dari matriks'''
    matrix = np.square(m)
    matrix = np.sum(matrix)
    matrix = np.sqrt(matrix)
    
    return matrix

def checkConverge(arr,arr_prev):
    ''' Mengecek nilai elemen matriks apakah konvergen menuju suatu nilai '''
    
    if (getNorm(arr_prev-arr)> 50000): # Parameter dapat diubah sesuai tingkat akurasi
            return False
    return True

def getEigenValueVector(A):
    ''' Algoritma QR untuk mencari eigenvalue dan eigenvector , dimana matriks Ak akan konvergen menuju matriks schur '''
    copyMatrix = np.copy(A)
    n = copyMatrix.shape[0]
    QQ = np.eye(n)
    i = 0
    while(True):
        copyMatrix_copy = np.copy(copyMatrix)
        s = copyMatrix.item(n-1, n-1)
        smult = s * np.eye(n)
        # Dengan metode gram schmidt dilcopyMatrixukan dekomposisi QR
        Q, R = gram_schmidt((copyMatrix - smult))
        # smult dikembalikan nilainya
        copyMatrix = np.add(R @ Q, smult)
        QQ = QQ @ Q
        i += 1
        if(checkConverge(copyMatrix,copyMatrix_copy)):
            # Cek parameter konvergen 
            break
    return copyMatrix,QQ # QQ adalah eigenvektor , copyMatrix adalah matrix yang diagonalnnya eigenvalue

def gram_schmidt(A):
    '''Algoritma QR menggunakan metode gram schmidt, mengeluarkan hasil dekomposisi matriks A'''  
    if len(A) != 0 :
        Q = np.empty([len(A), len(A)])
        counter = 0

        # Mencari Matrix Orthogonal
        for a in A.T:

            u = np.copy(a)
            for i in range(0, counter):
                proj = np.dot(np.dot(Q[:, i].T , a) , Q[:, i])
                u -= proj

            e = u / getNorm(u)
            Q[:, counter] = e

            counter += 1 

        # menghitung matriks segitiga atas
        R = np.dot(Q.T, A)

        return Q, R
    else:
        print("Image tidak kedetect")


def convertImage(imagename):
    '''Mengubah image menjadi matriks n*n x 1'''
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted


def convertFrame(frame):
    '''Mengubah Frame image menjadi matriks n*n x 1'''
    frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted


def getMatrixCoef(bestEigenVector, normalizedDataSet) :
    '''Menghasilkan List kombinasi linier dari normalized matriks yang ada pada dataset
        ,List ini berbentuk (<banyakeigenface>,<banyakgambar>)'''
    print("")
    CoefOfLinComMatrix = np.empty((len(bestEigenVector[0]),0), float)
    for i in range(len(normalizedDataSet[0])) :
        LinComOfNormalized = getCoef(bestEigenVector, np.transpose([normalizedDataSet[:,i]]))
        CoefOfLinComMatrix = np.column_stack((CoefOfLinComMatrix, LinComOfNormalized))
    return CoefOfLinComMatrix


def getCoef(list_bestEigenface, vectorImageInput) :
    '''Menghasilkan kombinasi linier dari normalized matriks, (<banyakeigenface>,1)'''
    matrixCoef = np.transpose([np.linalg.lstsq(list_bestEigenface, np.transpose(vectorImageInput)[0], rcond=None)[0]])
    return matrixCoef

def nearestDistance(InputCoef, MatrixCoef) :
    '''Mengeluarkan nilai X  minimum antara gambar input dengan gambar di data set'''
    '''X = |I-M|, I = Matriks kombinasi linear dari input dan M = List Matriks kombinasi linear dari gambar di dataset'''
    minimum = getNorm(np.subtract(InputCoef, np.transpose([MatrixCoef[:, 0]])))
    for i in range(len(MatrixCoef[0])) :
        e = getNorm(np.subtract(InputCoef, np.transpose([MatrixCoef[:, i]])))
        if (e < minimum) :
            minimum = e
    return minimum


def cropimage(image):
    '''Mengcrop gambar di wajah pengguna'''
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('src/src_feature/face.xml')
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
        
def cropframe(frame,path):
    '''Mengcrop gambar di wajah pengguna'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('src\src_feature\face.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h+10, x:x+w+10]
            cv2.imwrite(path, crop_img)


def list_eigenface(normalizedMatrix,all_image):
    '''Menghasilkan Matriks EigenFace dengan ukuran (256*256,<banyak eigenvector terbaik>)'''
    print("Mencari eigenface")
    Covariance = np.matmul(np.transpose(normalizedMatrix),normalizedMatrix) #A'A  = (<banyak gambar>, <banyak gambar>)
    eigenvalue, eigenvector = getEigenValueVector(Covariance)  
    bestEigenVector = np.empty((len(eigenvector[0]),0), float)
    counter = 0
    for i in range(len(eigenvector[0])) :
        if (eigenvalue[i][i] > 1) : # Tuning , Memilih eigenvector yang eigenvaluenya yang tidak koma
            bestEigenVector = np.column_stack((bestEigenVector, np.transpose([eigenvector[:, i]])))
            # break
        counter += 1
    # bestEigenVector = eigenvector[:, :counter]
    list_bestEigenface = np.empty((256*256, 0), float)
    for i in range(len(bestEigenVector[0])) :
        eigenface = np.matmul(all_image, np.transpose([bestEigenVector[:, i]])) 
        list_bestEigenface = np.column_stack((list_bestEigenface, eigenface))
    
    return list_bestEigenface
    #ini disave

def cropAllImage(path):
    # Mengcrop semua image
    print("Mengcrop semua image")
    folderpath = path
    print(folderpath)
    for (dirPath, dirNames, file) in os.walk(folderpath):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                cropimage(tempPath)
    print("Selesai crop semua image")



def outputImage (dirPath, MatrixCoef, InputCoef) :
    '''Mengeluarkan gambar yang paling mirip dengan gambar input di dataset'''
    minimum = getNorm(np.subtract(InputCoef, np.transpose([MatrixCoef[:, 0]])))
    imageOrder = 1
    for i in range(len(MatrixCoef[0])) :
        distance = getNorm(np.subtract(InputCoef, np.transpose([MatrixCoef[:, i]])))
        if (distance < minimum) :
            minimum = distance
            imageOrder = i + 1


    count = 0
    for (dirPath, dirNames, file) in os.walk(dirPath):
        for fileNames in file :
            count += 1
            if count == imageOrder :
                return os.path.join(dirPath, fileNames)


def trainingData(path):
    '''Menghasilkan matriks eigenface dan matriks koefisien'''
    folderpath = path
    A = np.empty((256*256,0), float) #Matriks Gambar (256*256,<Banyak Gambar>)
    for (dirPath, dirNames, file) in os.walk(folderpath):
        for fileNames in file :
                tempPath = os.path.join(dirPath, fileNames)
                convertedImage = convertImage(tempPath)
                A= np.column_stack((A, convertedImage.reshape(256*256, 1)))
    
    normalizedMatrix = A - A.mean(axis=1, keepdims=True)
    eigeface = list_eigenface(normalizedMatrix,A)
    MatrixCoef = getMatrixCoef(eigeface, normalizedMatrix)
    np.savetxt(f"src/data/matriksCoef.txt", MatrixCoef, delimiter=";")
    np.savetxt(f"src/data/eigenface.txt", eigeface, delimiter=";")
    print("Training Selesai")
    return MatrixCoef, eigeface


def closestImage(path, InputCoef, MatrixCoef):
    '''Mengluarkan Gambar yang paling mirip jika tidak ada keluarkan "Gambar tidak ditemukan" '''
    print("Mencari gambar terdekat")
    folderpath = path
    minimum = nearestDistance(InputCoef, MatrixCoef)
    print(minimum, "min")
    if minimum > 1.5 :   # Tuning minimum 
        print("Gambar terdekat")
        nearestImage =  outputImage(folderpath, MatrixCoef, InputCoef)
        print(nearestImage)
        return nearestImage
    else:
        nearestImage =  outputImage(folderpath, MatrixCoef, InputCoef)
        print(nearestImage)
        print("Gambar tidak ditemukan")
        return None


def displayEigenFace(eigenFace) :
    '''Menampilkan Gambar EigenFace'''
    print("Menampilkan eigenface")
    fig = plt.figure()
    for i in range(len(eigenFace[0])) :
        fig.add_subplot(10, 16, i+1)
        plt.imshow(eigenFace[:, i].reshape(256, 256), cmap='gray')
    plt.show()




