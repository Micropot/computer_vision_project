import sys
from src.NN import NN
import src.Parameters
import src.file_reading
import src.gui
from src.gui import Tk
from src.data_processing import ImageManagement

def main(argv):
    proj_param = src.Parameters.Parameters()
    proj_param = src.file_reading.YAML_Reader(argv[1], proj_param)

    #proj_param.Print()

    #******** TRAINING ********
    MyIA = NN()
    MyIA.load_dataset(proj_param)
    MyIA.CreateModel(proj_param)
   # MyIA.training(Parameters=proj_param)



    #*********** GUI ************
    root = Tk()
    p = src.gui.Draw(root, proj_param)
    #root.mainloop()


    #******** IMAGE **********
    MyImage = ImageManagement()

    MyImage.DonneeImage = MyImage.LectureFichierImage('/Users/arthurlamard/Documents/Allemagne/cours/computer_science/PROJECT/digit_reco/SaveImages/raw_image.png', proj_param)
    #MyImage.ResizeImage()





if __name__ == "__main__":
    main(sys.argv)