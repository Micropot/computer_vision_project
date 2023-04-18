# -*- coding: utf8 -*-
import cv2


class CaptureVideo() :
    def __init__(self):

       print "Debut CpatureVideo"
       self.cap = cv2.VideoCapture(0)
       print "capture Video"
       grayImage = None




    def run(self):

        ret, frame = self.cap.read()

        # Our operations on the frame come here
        self.grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', self.grayImage)

