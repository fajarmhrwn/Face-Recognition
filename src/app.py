import time
import tkinter
import tkinter.messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from eigenface import *
import customtkinter
import random
import string
import threading

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def createfilename():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def wait():
    for i in range(0,1000):
        print(i)


class App(customtkinter.CTk):
    image_camera = None
    Folder = None
    image_file = None
    WIDTH = 1600
    HEIGHT = 1200
    cam = None
    Frame = None
    img_counter = 0
    state = False
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=100,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=5)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(7, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Your Dataset",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose Folder",
                                                command=self.openFolder)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.label_2 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="No Input Folder Selected",
                                              text_font=("Roboto Medium", -10))  # font name and size in px
        self.label_2.grid(row=3, column=0, pady=10, padx=10)

        self.label_3 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Your Image",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_3.grid(row=4, column=0, pady=10, padx=10)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose File",
                                                command=self.openFile)
        self.button_3.grid(row=5, column=0, pady=10, padx=20)

        self.label_4 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="No Input File Selected",
                                              text_font=("Roboto Medium", -10))  # font name and size in px
        self.label_4.grid(row=6, column=0, pady=10, padx=10)

        self.switch_camera = customtkinter.CTkSwitch(master=self.frame_left,
                                                text="Camera: OFF",
                                                command=self.camera_event,
                                                text_font=("Roboto Medium", -10),
                                                onvalue="on", 
                                                offvalue="off")

        self.switch_camera.grid(row=8, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.label_time = customtkinter.CTkLabel(master=self.frame_left,
                                                text= f"Executed Time: 0 s",
                                                text_font=("Roboto Medium", -10))
        #left label_time
        self.label_time.grid(row=7, column=0, pady=10,padx=10)
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (8x4)
        self.frame_right.grid_rowconfigure(0, minsize=5)  # empty row with minsize as spacing
        self.frame_right.grid_rowconfigure((1,2,3,4,5,6,7,8,9), weight=1)  # empty row as spacing # empty row with minsize as spacing
        self.frame_right.grid_columnconfigure(0, weight=1)  # empty column as spacing
        self.frame_right.grid_columnconfigure(1, weight=1)  # empty column as spacing
        self.frame_right.grid_columnconfigure(2, weight=1)  # empty column as spacing
        self.frame_right.grid_columnconfigure(3, weight=1)  # empty column as spacing

        self.label_5 = customtkinter.CTkLabel(master=self.frame_right,
                                                text="Face Recognition",
                                                text_font=("Roboto Bold", -32))
        # center label_5 in frame_right
        self.label_5.grid(row=0, column=0, columnspan=4, pady=10, padx=10, sticky="nsew")

        self.label_title = customtkinter.CTkLabel(master=self.frame_right,text="Test Image",text_font=("Roboto Bold", -20))
        self.label_title.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

        self.label_title_1 = customtkinter.CTkLabel(master=self.frame_right,text="Closest Result",text_font=("Roboto Bold", -20))
        self.label_title_1.grid(row=1, column=2, columnspan=2, pady=10, padx=10, sticky="nsew")

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_right,
                                                    text="No Output Image",
                                                   height=300,
                                                   corner_radius=6,  # <- custom corner radius
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)

        # from row 1 to 8, column 2 to 3
        self.label_info_1.grid(row=2, column=2, rowspan=6, columnspan=2, pady=10, padx=10, sticky="nsew")
        self.label_info_2 = customtkinter.CTkLabel(master=self.frame_right,
                                                    text="No Image Selected",
                                                    height=300,
                                                   corner_radius=6,  # <- custom corner radius
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.RIGHT)
        # from row 1 to 8, column 0 to 1
        self.label_info_2.grid(row=2, column=0, rowspan=6, columnspan=2, pady=10, padx=10, sticky="nsew")
        self.button_camera = customtkinter.CTkButton(master=self.frame_right,
                                                text="Training Image",
                                                command=self.trainingImage)
        #beside near label_time
        self.button_camera.grid(row=8, column=0, pady=10, padx=20, sticky="w")                                        

    def camera_event(self):
        if self.switch_camera.get() == 'on':
            self.switch_camera.configure(text="Camera ON")
            self.button_1.configure(state=tkinter.DISABLED)
            self.button_3.configure(state=tkinter.DISABLED)
            self.opencamera()
            #open camera with MyvideoCapture
        else:
            self.switch_camera.configure(text="Camera OFF")
            self.button_1.configure(state=tkinter.NORMAL)
            self.button_3.configure(state=tkinter.NORMAL)
            self.stopcamera()

    def openFolder(self):
        folder = filedialog.askdirectory()
        App.Folder = folder
        print(folder)
        self.label_2.configure(text=folder[0:20] + "...")
        for f in os.listdir("src\data"):
            os.remove(os.path.join("src\data", f))
        trainingData(folder)


    
    def openFile(self):
        App.image_file = None

        self.label_time.configure(text="Executed Time : 0 s")
        file = filedialog.askopenfilename()
        image_path = file
        image_file = convertImage(image_path)
        image_file = image_file.reshape(256*256, 1)
        eigenface = np.loadtxt("src\data\eigenface.txt",delimiter=";")
        coefmatrix = np.loadtxt("src\data\matriksCoef.txt",delimiter=";")
        try:
            InputCoef = getCoef(eigenface, image_file)
            path = closestImage(App.Folder, InputCoef, coefmatrix)
            print(path)
            App.image_file = path
        except:
            App.image_file = None
            print("Dataset tidak ada")
        self.label_4.configure(text=file[0:20] + "...")
        self.img = Image.open(file)
        self.imgtk = ImageTk.PhotoImage(self.img.resize((256, 256)))
        self.label_info_2.configure(image=self.imgtk)
        self.label_info_2.image = self.imgtk
        self.outputimage()
            


    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

    def opencamera(self):
        self.label_time.configure(text="Executed Time : 0 s")
        self.label_2.configure(text="No Folder Selected")
        self.label_4.configure(text="No File Selected")
        App.cam = cv2.VideoCapture(0)
        App.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        App.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        App.cam.set(cv2.CAP_PROP_FPS, 30)
        self.everynseconds()
        while True:
            ret, frame = App.cam.read()
            App.Frame = frame
            #Update the image to tkinter...
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('src/src_feature/face.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            frame_tkinter=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_update = ImageTk.PhotoImage(Image.fromarray(frame_tkinter))
            #configure padding lael_info_2 to 0
            self.label_info_2.configure(image=img_update, padx=0, pady=0)
            self.label_info_2.image=img_update
            self.label_info_2.update()

            #every 10 seconds take a picture

            if not ret:
                print("failed to grab frame")
                break
        App.image_camera = None
        App.image_file = None

    def stopcamera(self):
        App.cam.release()
        cv2.destroyAllWindows()
        self.label_info_2.configure(image="")
        self.label_info_2.image=""
        self.label_info_2.update()
        App.image_camera = None
        App.image_file = None
        self.label_time.configure(text="Executed Time : 0 s")

    
    def keypressed(self, event):
        if event.char == '<Return>':
            print("Enter pressed")
            self.take_image()

    def take_imageInput(self):
        if App.Frame is not None:    
            img_camera = convertFrame(App.Frame)
            self.img = Image.fromarray(img_camera)
            self.imgtk = ImageTk.PhotoImage(self.img.resize((256, 256)))
            self.label_info_2.configure(image=self.imgtk)
            self.label_info_2.image = self.imgtk
            img_camera = img_camera.reshape(256*256, 1)
            matrixCoed,eigenface = trainingData("src/camera")
            try:
                InputCoef = getCoef(eigenface, img_camera)
                path = closestImage("src/camera", InputCoef, matrixCoed)
                print(path)
                App.image_camera = path
            except:
                App.image_camera = None
                print("Dataset tidak ada") 
            frame = App.Frame
            frame_tkinter=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_update = ImageTk.PhotoImage(Image.fromarray(frame_tkinter))
            self.label_info_2.configure(image=img_update, padx=0, pady=0)
            self.label_info_2.image=img_update
            self.label_info_2.update()       
            self.outputimage()

    def take_imageTraining(self):
        img_name = "src/camera/" + str(int(time.time())) + ".jpg"
        cv2.imwrite(img_name, App.Frame)
        cropimage(img_name)
        print("{} written!".format(img_name))
        frame = App.Frame
        frame_tkinter=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_update = ImageTk.PhotoImage(Image.fromarray(frame_tkinter))
        self.label_info_2.configure(image=img_update, padx=0, pady=0)
        self.label_info_2.image=img_update
        self.label_info_2.update()

    def trainingImage(self):
            start = time.time()
            while time.time() - start < 10:
                if (time.time() - start) % 0.5 == 0:
                    print("Take Image")
                    ret , frame = App.cam.read()
                    App.Frame = frame
                    self.take_imageTraining()
            
    def outputimage(self):
        #input image from camera or file
        #output image to label_info_2
        if App.image_camera is not None:
            self.img = Image.open(App.image_camera)
            resized_img = self.img.resize((500, 500), Image.ANTIALIAS)
            self.imgtk = ImageTk.PhotoImage(resized_img)
            start_time = time.time()
            self.label_info_1.configure(image=self.imgtk)
            self.label_info_1.image=self.imgtk
            self.label_info_1.update()
            self.label_time.configure(text="Executed Time: {:.2f} s".format(time.time() - start_time))
            print("output image from camera")
        elif App.image_file is not None:
            self.img = Image.open(App.image_file)
            self.imgtk = ImageTk.PhotoImage(self.img.resize((256, 256)))
            start_time = time.time()
            self.label_info_1.configure(image=self.imgtk)
            self.label_info_1.image=self.imgtk
            self.label_info_1.update()
            self.label_time.configure(text="Executed Time: {:.2f} s".format(time.time() - start_time))
            print("output image from file")
        else:
            self.label_info_1.configure(image="")
            self.label_info_1.image = None
            self.label_info_1.update()
            print("No image selected")
        
    def everynseconds(self):
        print("every 10 seconds")
        self.after(10000, self.everynseconds)
        self.take_imageInput()
    
if __name__ == "__main__":
    start = time.time()
    app = App()
    app.mainloop()

