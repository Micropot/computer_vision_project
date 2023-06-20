import sys
import os
from NN import NN
import src.Parameters
import src.file_reading
import src.SaveModels
import tensorflow as tf



def inference(argv):
    print("***** evalutaion ******")
    inf_param = src.Parameters.Parameters()
    inf_param = src.file_reading.YAML_Reader(argv[1], inf_param)
    user_path = os.path.join(inf_param.SaveModelDir, inf_param.wanted_model)
    model_path = os.path.join(user_path, "bestmodel.h5")

    MyIA_inf = NN()
    MyIA_inf.load_dataset(inf_param)
    MyIA_inf.CreateModel(inf_param)
    MyIA_inf.LoadModel(Parameters=inf_param)
    MyIA_inf.evaluate(inf_param)







if __name__ == "__main__":
    inference(sys.argv)