# -*- coding: utf8 -*-
import tensorflow as tf
mnist = tf.keras.datasets.mnist
#import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.utils import plot_model




class IA ():

    def __init__(self):
        print 'Construction de Mon IA'

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model = None
        self.prediction = -1



    def LectureDonnees(self):

        print "Lecture des données"
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        print "Normalisation des images entre 0 et 1"
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        print "Taille base de d'entrainement = ", len(self.x_train)

        print "Taille base de de test = ", len(self.x_test)


    def LanceApprentissage(self):

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=8)
        self.model.evaluate(self.x_test, self.y_test)



    def Prediction(self, MonImage):
        print "Prediction pour une image"

        result = self.model.predict(MonImage)

        print "result =", result

        self.prediction = result.argmax()

        print "prediction =", self.prediction

        #plt.title('Prediction: %d Label: %d' % (prediction, label))


    def SaveModel(self):
        print "SaveModel"
        self.model.save('/Users/Mathieu/Dev/Projets/LectureChiffre/models/mnistCNN.h5')



    def LoadModel(self):
        print "LoadModel"
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model = load_model('/Users/Mathieu/Dev/Projets/LectureChiffre/models/mnistCNN.h5')



    def VisualiseModel(self):
         plot_model(self.model, to_file='/Users/Mathieu/Dev/Projets/LectureChiffre/models/model.png')

