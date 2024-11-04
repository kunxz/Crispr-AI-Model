#data_loader.py

from tdc.single_pred import CRISPROutcome
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils import one_hot_encode

class CRISPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence = one_hot_encode(sequence)
        return (
            torch.tensor(encoded_sequence, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

def get_data_loaders(batch_size=32):
    #Specify a new data path
    data = CRISPROutcome(name='Leenay', label_name='Avg_Deletion_Length', path='./data/Leenay')
    splits = data.get_split()

    #Print the columns in splits['train']
    print("Columns in splits['train']:", splits['train'].columns.tolist())

    #Update column names based on actual data
    sequence_column = 'GuideSeq'  # Adjust based on printed columns
    label_column = 'Y'

    #Verify data
    print("First few rows of splits['train']:")
    print(splits['train'].head())


    #Create datasets
    train_dataset = CRISPRDataset(
        splits['train'][sequence_column], splits['train'][label_column]
    )
    valid_dataset = CRISPRDataset(
        splits['valid'][sequence_column], splits['valid'][label_column]
    )
    test_dataset = CRISPRDataset(
        splits['test'][sequence_column], splits['test'][label_column]
    )

    #Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader