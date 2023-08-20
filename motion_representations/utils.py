import os



def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)