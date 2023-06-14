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
        print ("Construction GestionImage")
        self.DonneeImage = None




    def LectureFichierImage(self, NomFichierImage, Parameters):
        print ("Lectrue du fichier :", NomFichierImage)
        img = Image.open(NomFichierImage).convert("L")
        img = img.resize((28, 28))
        print("type image : ",type(img))
        self.DonneeImage = np.array(img)
        print("type donnée image : ",type(self.DonneeImage))
        self.DonneeImage = self.DonneeImage.reshape(1,28,28)
        now = datetime.datetime.now()
        currentDateTime = now.strftime("%Y_%m_%d_%H_%M_%S")
        Parameters.image_path = os.path.join(Parameters.SaveImageDir, str(currentDateTime+'.png'))
        img.save(Parameters.image_path)
        #return(self.DonneeImage)

