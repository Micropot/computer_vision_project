# -*- coding: utf8 -*-
import datetime
import os

from PIL import Image
import numpy as np
import cv2
from skimage import data
from skimage import filters
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

class ImageManagement :
    def __init__(self):
        self.DonneeImage = None




    # read an image file, resize it and same the resized image
    def LectureFichierImage(self, NomFichierImage, Parameters):
        print ("Reading the file :", NomFichierImage)
        img = Image.open(NomFichierImage).convert("L")
        img = img.resize((28, 28))
        self.DonneeImage = np.array(img)
        self.DonneeImage = self.DonneeImage.reshape(1,28,28)
        now = datetime.datetime.now()
        currentDateTime = now.strftime("%Y_%m_%d_%H_%M_%S")
        Parameters.image_path = os.path.join(Parameters.SaveImageDir, str(currentDateTime+'.png'))
        img.save(Parameters.image_path)
        #return(self.DonneeImage)

