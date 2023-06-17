import numpy as np
import tensorflow as tf
import SaveModels
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.utils import to_categorical
import os
from PIL import Image
import PIL.ImageOps
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

mnist = tf.keras.datasets.mnist


class NN():
    def __int__(self):
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.model = None
        self.prediction = -1
        self.digit = None

    def load_dataset(self, Parameters):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        Parameters.x_test = self.x_test
        Parameters.y_test = self.y_test
        print("size of train : ", len(self.x_train))
        print("size of test : ", len(self.x_test))


    def CreateModel(self, Parameters):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer=Parameters.optimizer,
                           loss=Parameters.loss,
                           metrics=[Parameters.metrics])
        return self.model

    def training(self, Parameters):
        SaveModels.CreateFolder(Parameters, None)
        Parameters.current_model = self.model
        SaveModels.SaveModel(Parameters)

        Parameters.current_model.fit(
            self.x_train,
            self.y_train,
            batch_size=Parameters.batch_size,
            epochs=Parameters.epoch,
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            callbacks=[Parameters.cp_callback],
            verbose=2
        )

    def evaluate(self, Parameters):
        results = self.model.evaluate(Parameters.x_test, Parameters.y_test)
        print("test loss, test acc:", results)

    def Prediction(self, img):
        print("Prediction for one image")

        image = cv2.imread(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # make a rectangle box around each curve
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = th[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            digit = padded_digit.reshape(1, 28, 28, 1)
            self.digit = digit / 255.0

        res = self.model.predict([self.digit])[0]
        print("resut : ", res)
        self.prediction = res.argmax()
        print("prediction =", self.prediction)
        data = str(self.prediction) + ' ' + str(int(max(res) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyWindow('image')
        return np.argmax(res), max(res)

    def LoadModel(self, Parameters):
        user_path = os.path.join(Parameters.SaveModelDir, Parameters.wanted_model)
        model_path = os.path.join(user_path, "bestmodel.h5")
        print(model_path)
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model = load_model(model_path)
