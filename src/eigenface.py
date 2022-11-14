# Algoritma eigenface , ngitung eigenvalue pake svd
import numpy as np
import os
import face_detection

def checkDiagonal(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                continue
            else:
                if abs(arr[i][j]) > 0.001 and (i > j):
                    return False
    return True

def qrFactorization(arr):
    temp = arr
    i = 0
    while(True):
        Q,R = np.linalg.qr(temp)
        temp = np.dot(R, Q)
        if(checkDiagonal(temp)):
            print("Number of Factorizations: " + str(i+1))
            break
        else:
            i += 1
    
        # print(temp)

    
    return temp

def getEigenvalue(arr):
    temp = np.zeros(len(arr),1)
    count = 1
    for i in range(len(arr)):
        temp[i][0] = arr[i][i]
        if(abs(temp[i][0]) < 0.000000000001):
            temp = 0
        print("Lamda"+str(count) +": " + str(temp[i][0]))
        count += 1

    return temp
    
# def read():
#     f = open('src\matrix.txt', 'r')
#     temp = f.read().split('\n')
#     arr = []
#     for i in temp:
#         if i == '':
#             continue
#         arr.append(i.split(" "))
#     for i in range(len(arr)):
#         for j in range(len(arr[i])):
#             arr[i][j] = int(arr[i][j])
#     return arr

def eigenface(A,N,pth):
    
    
    
    path = r"test/pins_dataset/" + pth #"pins_dataset" itu nama foldernya
    temp = os.listdir(path)
    eigval = getEigenvalue(qrFactorization(A))
    eigenfaceMat = np.zeros(len(A),len(A))
    i = 0
    for file in temp:
        a = face_detection.convertImage("test/pins_dataset/"+pth+"/"+file)
        a = np.reshape(a,(256,256))

        temp = np.zeros(len(A),len(A[0]))
        eigvalMat = np.fill_diagonal(temp,eigval[i])
        eigvalMat = np.subtract(eigvalMat,A)
        b = np.zeros(len(A))
        eigvector = np.linalg.solve(eigvalMat, b)
        eigenfaceMat = np.add(eigenfaceMat,np.matmul(eigvector,a))

    return eigenfaceMat

# if __name__ == '__main__':
#     main()