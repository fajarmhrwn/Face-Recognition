from eigenface import *
# cropimage("test/input/test_fajar.png")
print("Mulai")
path = input("Masukkan nama folder dataset: ")
# cropAllImage(path)
coefmatrix, eigenface = trainingData("test/pins_dataset/" + path)
displayEigenFace(eigenface)
image_path = input("Masukkan nama file gambar: ")
Image = convertImage("test/input/"+image_path)
Image = Image.reshape(256*256, 1)
InputCoef = getCoef(eigenface, Image)
path = closestImage(path, InputCoef, coefmatrix)
#show image
if path != None:
    img = cv2.imread(path)
    cv2.imwrite("test/input/1"+image_path, img)