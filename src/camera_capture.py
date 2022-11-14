import cv2

cascPath = "src\src_features\\face.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    rect, frame = video_capture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the facesq
    # print(faces)
    for (x, y, w, h) in faces:        
        cv2.rectangle(frame, (x, y), (x+w, y+h), ( 255, 0, 0), 2)        
        # center_coordinates = x + w // 2, y + h // 2
        # radius = w // 2 # or can be h / 2 or can be anything based on your requirements
        # cv2.circle(frame, center_coordinates, radius, (0, 0, 100), 3)
       

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    elif (cv2.waitKey(1) == ord('c')): # press c to capture image and save it to src
        roi = frame[y:y+h, x:x+w]
        face_name = input("Enter the name of the person: ")
        cv2.imwrite("src\\camera_image\\"+ face_name +".png", roi)

        
        img_resize = cv2.imread("src\\camera_image\\"+ face_name +".png", cv2.IMREAD_UNCHANGED)


        # print('Original Dimensions : ',img_resize.shape)
        
        # scale_percent = 60 # percent of original size
        width = int(256)
        height = int(256)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img_resize, dim, interpolation = cv2.INTER_AREA)
        print('Original Dimensions : ',roi.shape)
        print('Resized Dimensions : ',resized.shape)
        cv2.imwrite("src\\camera_image\\"+ face_name +"_resized.png", resized)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()