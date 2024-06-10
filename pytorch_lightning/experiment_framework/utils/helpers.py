# Imports
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from utils.plots import *   

def get_latest_version(path):
    """
    Purpose:
    - Gather most recent version folder
    
    Arguments:
    - path (str): path to version folders
    """

    all_folders = [int(ele.replace("version_", "")) for ele in os.listdir(path)]

    return all_folders[np.argmax(all_folders)]

def get_training_results(path, path_plots, target_names, show_plots, save_plots):
    """
    Purpose:
    - Gather and show training analytics
    
    Arguments:
    - path (str): path to analytics file
    - target_names (list[str]): columns of analytics file to display
    """

    # Gather: All Analytics
    
    print("Loading path: %s" % path)

    data = pd.read_csv(path)

    # Display: Target Analytics
    
    for name in target_names:
    
        df = data.dropna(subset=[name])
        
        if "lr" in name:
            tag = "epoch"
            x_vals = list(range(df.shape[0]))
        else:
            tag = name.split("_")[-1]
            x_vals = df[tag]
        
        y_vals = df[name]
    
        title = "Plotting %s vs %s" % (name, tag)
        y_label = "%s" % name
        x_label = "%s" % tag
    
        plot_training(x_vals, y_vals, title, x_label, y_label, path_plots, show_plots, save_plots)