import glob
import os.path
import sys
from src.NN import NN
import src.Parameters
import src.file_reading
import src.gui
from src.gui import Tk
from src.data_processing import ImageManagement

#TODO : https://medium.com/analytics-vidhya/handwritten-digit-recognition-gui-app-46e3d7b37287
def main(argv):
    proj_param = src.Parameters.Parameters()
    proj_param = src.file_reading.YAML_Reader(argv[1], proj_param)

    #proj_param.Print()

    #******** TRAINING ********
    MyIA = NN()
    MyIA.load_dataset(proj_param)
    MyIA.CreateModel(proj_param)
    #MyIA.training(Parameters=proj_param)
    MyIA.LoadModel(proj_param)


    #*********** GUI ************
    root = Tk()
    p = src.gui.Draw(root, proj_param)
    root.mainloop()


    #******** IMAGE PREDICTION**********
    '''MyImage = ImageManagement()

    MyImage.DonneeImage = MyImage.LectureFichierImage(str(proj_param.SaveImageDir + 'raw_image.png'), proj_param)
    #MyImage.ResizeImage()
    list_of_file = glob.glob(str(proj_param.SaveImageDir + '/*.png'))
    latest_file = max(list_of_file, key=os.path.getctime)
    print("latest file : ", latest_file)
    #MyIA.Prediction(latest_file)
    MyIA.Prediction(str(proj_param.SaveImageDir + 'raw_image.png'))'''





if __name__ == "__main__":
    main(sys.argv)