import glob
import os.path
import sys

from custom_model import scan_photo
from src.NN import NN
import src.Parameters
import src.file_reading
import src.gui
from src.gui import Tk
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog

from src.data_processing import ImageManagement

def save_prediction(correct_prediction, prediction_name):
    if correct_prediction:
        filename = str(prediction_name) + ".txt"
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=filename)
        if file_path:
            with open(file_path, "w") as file:
                file.write(str(prediction_name))
                messagebox.showinfo("Success", "Prediction saved successfully!")
        else:
            messagebox.showinfo("Cancelled", "Prediction not saved.")

def display_popup(prediction_name):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    message = "Is the prediction '{}' correct?".format(prediction_name)
    confirmed = messagebox.askyesno("Confirmation", message)
    if not confirmed:
        correct_prediction = simpledialog.askstring("Correct Prediction", "Please enter the correct prediction:")
        if correct_prediction:
            save_prediction(True, correct_prediction)

    save_prediction(confirmed, prediction_name)

def button_click():
    print("Button clicked!")
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)
    global pred
    pred = scan_photo(file_path)


#TODO : https://medium.com/analytics-vidhya/handwritten-digit-recognition-gui-app-46e3d7b37287
def main(argv):
    proj_param = src.Parameters.Parameters()
    proj_param = src.file_reading.YAML_Reader(argv[1], proj_param)

    # proj_param.Print()

    #******** TRAINING ********
    MyIA = NN()
    MyIA.load_dataset(proj_param)
    MyIA.CreateModel(proj_param)
    #MyIA.training(Parameters=proj_param)
    MyIA.LoadModel(proj_param)


    #*********** GUI ************
    root = Tk()
    p = src.gui.Draw(root, proj_param)
    button = tk.Button(root, text="Upload photo", command=button_click)
    button.pack()

    root.mainloop()


    #******** IMAGE PREDICTION **********
    '''MyImage = ImageManagement()

    MyImage.DonneeImage = MyImage.LectureFichierImage(str(proj_param.SaveImageDir + 'raw_image.png'), proj_param)
    #MyImage.ResizeImage()
    list_of_file = glob.glob(str(proj_param.SaveImageDir + '/*.png'))
    latest_file = max(list_of_file, key=os.path.getctime)
    print("latest file : ", latest_file)
    #MyIA.Prediction(latest_file)
    MyIA.Prediction(str(proj_param.SaveImageDir + 'raw_image.png'))'''

    #******** POP UP WINDOW WITH FILE SAVING **********
    display_popup(pred)




if __name__ == "__main__":
    main(sys.argv)
