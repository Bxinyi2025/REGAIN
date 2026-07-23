import os
import datetime as dt
import json
import collections
import re
import torch
from scipy.special import softmax
import numpy as np
import random
import matplotlib.pyplot as plt
import natsort
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import tifffile
import scipy
import pandas as pd
import glob
import h5py
from sklearn.manifold import TSNE


def get_device(gpu_id):
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print("Using GPUs: {}".format(gpu_str))
    device = torch.device("cuda")

    return device


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def read_txt(fp):
    with open(fp) as file:
        lines = [line.rstrip() for line in file]
    return lines


def delete_file(path):
    """
    Delete file if exists
    """
    if os.path.exists(path):
        os.remove(path)


def get_files_list(path, ext_array=[".tif"]):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_newest_id(exp_dir="experiments", fold_id=1):
    """Get the latest experiment ID based on its timestamp

    Parameters
    ----------
    exp_dir : str, optional
        Name of the directory that contains all the experiment directories, by default 'experiments'

    Returns
    -------
    exp_id : str
        Name of the latest experiment directory
    """
    folders = next(os.walk(exp_dir))[1]
    folders = natsort.natsorted(folders)
    # folders = [x for x in folders if mode in x]
    folders = [x for x in folders if ("fold" + str(fold_id) + "_") in x]
    folder_last = folders[-1]
    exp_id = folder_last.replace("\\", "/")
    return exp_id


def get_experiment_id(make_new, load_dir, fold_id):
  
    return  load_dir


def is_valid_positive_int(s):
    try:
        return int(s) > 0
    except ValueError:
        return False
