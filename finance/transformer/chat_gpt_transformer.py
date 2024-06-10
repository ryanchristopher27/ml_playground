import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Data Preprocessing
class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_len], self.data[index+1:index+self.seq_len+1])

def load_data(file_path, seq_len):
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    data = scaler.fit_transform(df[['Close']].values)
    data = torch.FloatTensor(data)
    dataset = StockDataset(data, seq_len)
    return DataLoader(dataset, batch_size=64, shuffle=True), scaler

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        memory = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.fc_out(memory)
        return output

# Training Loop
def train(model, dataloader, num_epochs=10, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for src, tgt in dataloader:
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Example usage
if __name__ == "__main__":
    seq_len = 32  # Sequence length
    data_loader, scaler = load_data('path_to_your_stock_data.csv', seq_len)

    # Instantiate the model
    input_dim = 1  # Since we are only using the 'Close' price
    model = TransformerModel(input_dim)

    # Train the model
    train(model, data_loader)
