import sys
from src.NN import NN
import src.Parameters
import src.file_reading
import src.gui
from src.gui import Tk

def main(argv):
    #TODO requiement environement for git hub ok c'est bon
    proj_param = src.Parameters.Parameters()
    proj_param = src.file_reading.YAML_Reader(argv[1], proj_param)

    #proj_param.Print()

    #******** TRAINING ********
    MyIA = NN()
    MyIA.load_dataset(proj_param)
    MyIA.CreateModel(proj_param)
    print("--------- TYPE :", type(MyIA.CreateModel(proj_param)))
    #MyIA.training(Parameters=proj_param)
    MyIA.LoadModel(proj_param)
    MyIA.evaluate(proj_param)


    #*********** GUI ************
    root = Tk()
    p = src.gui.Draw(root)
    #root.mainloop()






if __name__ == "__main__":
    main(sys.argv)