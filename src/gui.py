import datetime
import glob
from tkinter import *
import tkinter as tk
from tkinter.ttk import Scale
from tkinter import colorchooser, filedialog, messagebox
import subprocess
import os
import PIL.ImageGrab as ImageGrab

import SaveImage
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import simpledialog
from pathlib import Path

from NN import NN
from data_processing import ImageManagement


#import Parameters
#import SaveImage


# Defining Class and constructor of the Program
class Draw:
    def __init__(self, root, Parameters):
        # Defining title and Size of the Tkinter Window GUI
        self.Parameters = Parameters
        self.root = root
        self.root.title("Copy Assignment Painter")
        self.root.geometry("810x530")
        self.root.configure(background="white")
        #         self.root.resizable(0,0)

        # variables for pointer and Eraser
        self.pointer = "black"
        self.erase = "white"
        self.background = "white"
        self.last_x = None
        self.last_y = None

        # Widgets for Tkinter Window

        # Configure the alignment , font size and color of the text
        text = Text(root)
        text.tag_configure("tag_name", justify='center', font=('arial', 25), background='#292826', foreground='orange')

        # Insert a Text
        text.insert("1.0", "Handwritten digits detector")

        # Add the tag for following given text
        text.tag_add("tag_name", "1.0", "end")
        #text.pack()

        # Pick a color for drawing from color pannel
        self.pick_color = LabelFrame(self.root, text='Colors', font=('arial', 15), bd=5, relief=RIDGE, bg='white')
        self.pick_color.place(x=0, y=40, width=90, height=185)

        colors = ['blue', 'red', 'green', 'orange', 'violet', 'black', 'yellow', 'purple', 'pink', 'gold', 'brown',
                  'indigo']
        i = j = 0
        for color in colors:
            Button(self.pick_color, bg=color, bd=2, relief=RIDGE, width=3,
                   command=lambda col=color: self.select_color(col)).grid(row=i, column=j)
            i += 1
            if i == 6:
                i = 0
                j = 1

        # Erase Button and its properties
        self.eraser_btn = Button(self.root, text="Eraser", bd=4, bg='white', command=lambda: self.select_color(self.erase), width=9, relief=RIDGE) #self.select_color(self.erase)
        self.eraser_btn.place(x=0, y=227)

        # Reset Button to clear the entire screen
        self.clear_screen = Button(self.root, text="Clear Screen", bd=4, bg='white',
                                   command=lambda: self.background.delete('all'), width=9, relief=RIDGE)
        self.clear_screen.place(x=0, y=257)

        # Save Button for saving the image in local computer
        self.save_btn = Button(self.root, text="Predict", bd=4, bg='white', command=self.save_drawing, width=9,
                               relief=RIDGE)
        self.save_btn.place(x=0, y=287)

        # Background Button for choosing color of the Canvas
        #self.bg_btn = Button(self.root, text="Background", bd=4, bg='white', command=self.canvas_color, width=9,relief=RIDGE)
        #self.bg_btn.place(x=0, y=317)

        # Creating a Scale for pointer and eraser size
        self.pointer_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'), relief=RIDGE)
        self.pointer_frame.place(x=0, y=350, height=170, width=70)

        self.pointer_size = Scale(self.pointer_frame, orient=VERTICAL, from_=48, to=100, length=148)
        self.pointer_size.set(1)
        self.pointer_size.grid(row=0, column=1, padx=15)


        # Defining a background color for the Canvas
        self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=470, width=680)
        self.background.place(x=80, y=40)

        # Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>", self.paint)


    # Functions are defined here
    # Paint Function for Drawing the lines on Canvas
    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)

        '''self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer,
                                    width=self.pointer_size.get())'''
        self.background.create_line((x1,y1,x2,y2),width=8, fill=self.pointer, capstyle=ROUND, smooth=TRUE, splinesteps=12)

    def SaveFile(self, Parameters):
        dir_path = os.path.dirname(Parameters.SaveImageDir)
        print("dir_path : ",dir_path)


    # Function for saving the image file in Local Computer
    def save_drawing(self):
        try:
            self.background.update()
            #Create a folder for the images
            SaveImage.CreateFolder_Image(self.Parameters)
            file_ss = 'raw_image.png'
            image_path = os.path.join(self.Parameters.SaveImageDir, file_ss)
            x = self.root.winfo_rootx() + self.background.winfo_x() + 10
            y = self.root.winfo_rooty() + self.background.winfo_y() + 10
            x1 = x + self.background.winfo_width() - 20
            y1 = y + self.background.winfo_height() - 20
            # take a screenshot of the screen and resize it
            ImageGrab.grab((x, y, x1, y1)).save(image_path)

            #messagebox.showinfo('Screenshot Successfully Saved as' + str(file_ss))

            MyImage = ImageManagement()
            MyIA = NN()
            MyIA.LoadModel(self.Parameters)

            MyImage.DonneeImage = MyImage.LectureFichierImage(str(self.Parameters.SaveImageDir + 'raw_image.png'),
                                                              self.Parameters)
            # MyImage.ResizeImage()
            list_of_file = glob.glob(str(self.Parameters.SaveImageDir + '/*.png'))
            latest_file = max(list_of_file, key=os.path.getctime)
            print("latest file : ", latest_file)
            # MyIA.Prediction(latest_file)

            MyIA.Prediction(str(self.Parameters.SaveImageDir + 'raw_image.png'), self.Parameters)
            SaveImage.CreateFolder_label(self.Parameters)
            USER_INP = simpledialog.askstring(title="Test",
                                              prompt="Is your prediction correct (y/n) ? :")
            if USER_INP == 'y':
                print("YES")
                pass
            elif USER_INP == "n":
                print("NO")
                USER_LABEL = simpledialog.askstring(title="Label",prompt="What was your number ? :")
                print("USER_LABEL : ",USER_LABEL)
                label = os.path.basename(os.path.normpath(self.Parameters.image_path))
                new_label = os.path.splitext(label)[0]
                label_path = os.path.join(self.Parameters.LabelsDir, str(new_label+'.txt'))
                #print(label_path)
                with open(label_path, 'w') as f:
                    f.write(USER_LABEL)
                f.close()
            else:
                print("Error, please enter y or n ")

        except:
            print("Error in saving the saving")


