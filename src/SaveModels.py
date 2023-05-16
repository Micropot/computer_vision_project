import datetime
import os
import tensorflow as tf
from keras.utils import plot_model


def CreateFolder(Parametre, trial):
    dir_path = os.path.dirname(Parametre.SaveModelDir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    now = datetime.datetime.now()
    currentDateTime = now.strftime("%Y_%m_%d_%H_%M_%S")

    if trial is None:
        folder_path = os.path.join(dir_path, currentDateTime)
        os.mkdir(folder_path)
        Parametre.SaveModelDir = folder_path


def SaveModel(Parametres):
    checkpoint_path = os.path.join(Parametres.SaveModelDir, "bestmodel.h5")
    Parametres.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    Parametres.cp_callback = [Parametres.checkpoint]

def VisualiseModel(Parameters):
    plot_path = os.path.join(Parameters.SaveModelDir, 'model.png')
    plot_model(Parameters.current_model, to_file=plot_path)


