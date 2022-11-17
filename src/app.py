import tkinter
import tkinter.messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    WIDTH = 1600
    HEIGHT = 1200
    time = 0
    cam = None
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

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.rowconfigure(1, weight=1)
        self.frame_info.columnconfigure(0, weight=1)
        self.frame_info.columnconfigure(1, weight=1)
        
        self.label_5 = customtkinter.CTkLabel(master=self.frame_info,
                                                text="Face Recognition",
                                                text_font=("Roboto Bold", -32))
        self.label_5.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                    text="No Image Selected",
                                                   height=300,
                                                   corner_radius=6,  # <- custom corner radius
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)

        self.label_info_1.grid(column=0, row=1, sticky="nwe", padx=15, pady=15)

        self.label_info_2 = customtkinter.CTkLabel(master=self.frame_info,
                                                    text="No Output Image",
                                                    height=300,
                                                   corner_radius=6,  # <- custom corner radius
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_2.grid(column=1, row=1, sticky="nwe", padx=15, pady=15)

        self.label_time = customtkinter.CTkLabel(master=self.frame_right,
                                                text= f"Executed Time: {self.time}s",
                                                text_font=("Roboto Medium", -10))
        #left label_time
        self.label_time.grid(row=4, column=0, pady=10, padx=20, sticky="w")
    def camera_event(self):
        if self.switch_camera.get() == 'on':
            self.switch_camera.configure(text="Camera ON")
            self.opencamera()
            #open camera with MyvideoCapture
        else:
            self.switch_camera.configure(text="Camera OFF")
            self.stopcamera()

    def openFolder(self):
        folder = filedialog.askdirectory()
        print(folder)
        self.label_2.configure(text=folder[0:20] + "...")
    
    def openFile(self):
        file = filedialog.askopenfilename()
        print(file)
        self.label_4.configure(text=file[0:20] + "...")
        self.img = Image.open(file)
        self.imgtk = ImageTk.PhotoImage(self.img.resize((256, 256)))
        self.label_info_1.configure(image=self.imgtk)

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

    def opencamera(self):
        frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        App.cam = cv2.VideoCapture(0)
        #cv2.namedWindow("Experience_in_AI camera")
        while True:
            ret, frame = App.cam.read()

            #Update the image to tkinter...
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_update = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label_info_1.configure(image=img_update)
            self.label_info_1.image=img_update
            self.label_info_1.update()

            if not ret:
                print("failed to grab frame")
                break

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")

                App.cam.release()
                cv2.destroyAllWindows()
                break
    def stopcamera(self):
        App.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.mainloop()