import os

def CreateFolder_Image(Parametre):
    dir_path = os.path.dirname(Parametre.SaveImageDir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def CreateFolder_label(Parameters):
    label_dir_path = os.path.join(Parameters.PROJECT_DIR,"Labels")
    Parameters.LabelsDir = label_dir_path
    if not os.path.exists(label_dir_path):
        os.mkdir(label_dir_path)

