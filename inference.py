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
    model_path = os.path.join(user_path, "bestmodel.hdf5")

    MyIA_inf = NN()
    #MyIA_inf.load_dataset(inf_param)
    #inf_param.inf_model = MyIA_inf.CreateModel(inf_param)
    inf_model = MyIA_inf.LoadModel(Parameters=inf_param)
    inf_model.load_weights(model_path)

    #results = inf_model.evaluate(inf_param.x_test, inf_param.y_test)
    #print("test loss, test acc:", results)





if __name__ == "__main__":
    inference(sys.argv)