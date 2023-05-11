import tensorflow as tf
import SaveModels
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import os

mnist = tf.keras.datasets.mnist

class NN():
    def __int__(self):
        print("NN construction ")
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.model = None
        self.prediction = -1


    def load_dataset(self, Parameters):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess the data (these are NumPy arrays)
        '''self.x_train = self.x_train.reshape(60000, 784).astype("float32") / 255
        self.x_test = self.x_test.reshape(10000, 784).astype("float32") / 255'''
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.y_train = self.y_train.astype("float32")
        self.y_test = self.y_test.astype("float32")
        Parameters.x_test = self.x_test
        Parameters.y_test = self.y_test

        # Reserve 10,000 samples for validation
        self.x_val = self.x_train[-10000:]
        self.y_val = self.y_train[-10000:]
        self.x_train = self.x_train[:-10000]
        self.y_train = self.y_train[:-10000]

        print("size of train : ", len(self.x_train))
        print("size of validation : ", len(self.x_val))
        print("size of test : ", len(self.x_test))

    def CreateModel(self, Parameters):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=Parameters.optimizer,
                           loss=Parameters.loss,
                           metrics=[Parameters.metrics])



    def training(self, Parameters):
        SaveModels.CreateFolder(Parameters, None)

        #self.model = self.CreateModel(Parameters)
        Parameters.current_model = self.model
        SaveModels.SaveModel(Parameters)

        Parameters.current_model.fit(
            self.x_train,
            self.y_train,
            batch_size=Parameters.batch_size,
            epochs=Parameters.epoch,
            validation_data=(self.x_val, self.y_val),
            shuffle=True,
            callbacks=[Parameters.cp_callback]
        )
        #SaveModels.VisualiseModel(Parameters)

    def evaluate(self, Parameters):
        results = self.model.evaluate(Parameters.x_test, Parameters.y_test)
        print("test loss, test acc:", results)



    def Prediction(self, MyImage):
        print ("Prediction for one image")

        result = self.model.predict(MyImage)

        print ("result =", result)

        self.prediction = result.argmax()

        print ("prediction =", self.prediction)

    def LoadModel(self, Parameters):
        user_path = os.path.join(Parameters.SaveModelDir, Parameters.wanted_model)
        model_path = os.path.join(user_path, "bestmodel.hdf5")
        print(model_path)
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model = load_model(model_path)
