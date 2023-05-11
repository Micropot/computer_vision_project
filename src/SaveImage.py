import os

def CreateFolder_Image(Parametre):
    dir_path = os.path.dirname(Parametre.SaveImageDir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
