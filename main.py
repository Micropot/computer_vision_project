# -*- coding: utf8 -*-


#Développment d'une intéllige,nce artificielle pour lire des chiffres.
from IA import IA
from GestionImage import GestionImage
from CaptureVideo import CaptureVideo
import cv2

MonIA =IA()
#MonIA.LectureDonnees()
#MonIA.LanceApprentissage()
#MonIA.SaveModel()
print "CaptureVideo"



MaCaptureVideo = CaptureVideo()


MonIA.LoadModel()

# Capture frame-by-frame


Continue = True

while Continue:

    while (True):
        MaCaptureVideo.run()
        k = cv2.waitKey(1)
        if k == ord('q') :
            break
        if k == ord('s'):
            Continue = False
            break

    #MonIA.VisualiseModel()
    #Lecture d'une image de test :
    if (Continue):
        MonImage = GestionImage()
        MonImage.DonneeImage= MaCaptureVideo.grayImage
        #MonImage.LectureFichierImage("/Users/Mathieu/Dev/Projets/LectureChiffre/testSample/img_3.jpg")
        MonImage.ResizeImage()

        MonIA.Prediction(MonImage.DonneeImage)


