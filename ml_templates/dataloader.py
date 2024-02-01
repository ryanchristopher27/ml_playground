import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class DatasetTemplate(Dataset):
    def __init__(self):
        # Data Loading
        data = np.loadtxt('./path/to/file', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = data[:, 1:]
        self.y = data[:, [0]]
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
# Data Loader
dataset = DatasetTemplate()
data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)

# Data Iterator
data_iter = iter(data_loader)
data = data_iter.next()
features, labels = data
print(features, labels)


# Training Loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        if (i+1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape}')
        # Forward Pass

        # Backward Pass

        # Update Weights
