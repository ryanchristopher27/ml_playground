# Imports
import torch

def cuda_setup() -> ():
    if torch.cuda.is_available():
        print(torch.cuda.current_device())     # The ID of the current GPU.
        print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
        print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
        print(torch.cuda.device_count())
        
    on_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    return device, on_gpu


# Imports
import seaborn as sn  # yes, I had to "conda install seaborn"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes, num_samples, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    ConfusionMatrix = torch.zeros((num_classes ,num_classes))
    for i in range(num_samples):
        ConfusionMatrix[y_true,y_pred] = ConfusionMatrix[y_true,y_pred] + 1

    df_cm = pd.DataFrame(np.asarray(ConfusionMatrix), index = [i for i in class_names],
                    columns = [i for i in class_names])
    
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()    