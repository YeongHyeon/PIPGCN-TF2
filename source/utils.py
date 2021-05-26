import os, glob, shutil, pickle, gzip

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def read_pickle(path):

    with gzip.open(path, 'rb') as f:
        elm0, elm1 = pickle.load(f, encoding='latin1')
    return elm0, elm1
