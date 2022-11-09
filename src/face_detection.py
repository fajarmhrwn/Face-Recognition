# pake opencv buat ngambil wajah dr kamera
# dilanjut sampe konversi greyscale, kebentuk matriks 

import cv2

def convertedImage(imagename):
    image = cv2.imread(imagename)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    greyscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    converted = greyscaleimg.flatten()
    return converted