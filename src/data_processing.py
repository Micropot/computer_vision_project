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

    def getBestShift(self,img):
        cy, cx = ndimage.measurements.center_of_mass(img)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty


    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted


    def ResizeImage(self):
        print ("Inversion Image")

        self.DonneeImage = 255 - self.DonneeImage

        print ("Trouver Seuil Image")

        val = filters.threshold_otsu(self.DonneeImage)
        print ("seuil =", val)
        #self.DonneeImage = self.DonneeImage.resize(28,28)

        print ("seuillage")

        self.DonneeImage[self.DonneeImage < val*1.20] = 0

        cv2.imshow('Test-1', self.DonneeImage)

        print ("Recherche de la boundiong box")

        a = np.where(self.DonneeImage != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])



        print (bbox)

        print ("crop the image")
        self.DonneeImage = self.DonneeImage[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        gray = self.DonneeImage
        rows, cols = gray.shape
        compl_dif = abs(rows - cols)
        half_Sm = compl_dif / 2
        half_Sm = int(half_Sm)
        half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
        half_Big = int(half_Big)
        if rows > cols:
            gray = np.lib.pad(gray, ((0, 0), (half_Sm, half_Big)), 'constant', constant_values=0)
        else:
            gray = np.lib.pad(gray, ((half_Sm, half_Big), (0, 0)), 'constant', constant_values=0)

        print ("pad with 0 pour la rendre carré")

        gray = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_CUBIC)
        gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')



        self.DonneeImage = np.array(gray)
        self.DonneeImage = self.DonneeImage.reshape(1, 28, 28)
