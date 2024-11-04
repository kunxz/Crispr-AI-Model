#main.py

from data_loader import get_data_loaders
from model import CRISPRCNN  # or CRISPRCNN if using CNN
from train import train_model
from evaluate import evaluate_model
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    #Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    seeds = [55, 123, 552, 789, 852]
    mse_list = []
    mae_list = []
    r2_list = []

    #Get data loaders
    train_loader, valid_loader, test_loader = get_data_loaders(batch_size=32)

    for seed in seeds:
        print(f"\nRunning with seed {seed}")
        set_seed(seed)

        #Initialize model
        model = CRISPRCNN(sequence_length=23)  # or CRISPRCNN if using CNN
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Train the model
        train_model(model, train_loader, criterion, optimizer, num_epochs)

        #Evaluate the model
        mse, mae, r2 = evaluate_model(model, valid_loader)

        print(f'Seed {seed} - MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}')
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

    #Compute average and standard deviation
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    avg_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)

    print(f'\nAverage MSE over runs: {avg_mse:.4f} ± {std_mse:.4f}')
    print(f'Average MAE over runs: {avg_mae:.4f} ± {std_mae:.4f}')
    print(f'Average R^2 over runs: {avg_r2:.4f} ± {std_r2:.4f}')

if __name__ == '__main__':
    main()