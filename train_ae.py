import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import statistics as stat
from sklearn.preprocessing import StandardScaler
import copy
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from dataset import *
from sklearn.model_selection import train_test_split
import torch.optim as optim
import polars as pl
import random
import gc
from utils import *
from models import *

model_save_dir = '/home/siddharth/jane_street_challenge_2024/models'
jane_street_real_time_market_data_forecasting_path = '/home/siddharth/jane_street_challenge_2024/data'
load_model = True
evaluate = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder()
model = model.to(device)
model_name = "encoder_v2"
# load model if present in dir and flag is set
if load_model:
    model.load_state_dict(torch.load(f'{model_save_dir}/{model_name}.pth',weights_only=True))
    print("Model loaded")

optimizer = optim.Adam(model.parameters(), lr=1e-4) #Adam optimizer
loss_function = nn.MSELoss(reduction='none')
epochs = 30
best = float('inf')
degraded = 0
batch_size = 4096

alltraindata = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")
# Get unique date_ids greater than 1400 and shuffle them
unique_date_ids = alltraindata.select("date_id").unique().collect().to_series().to_list()
test_date_ids_specific = [date_id for date_id in unique_date_ids if 1660 <= date_id <= 1698]
unique_date_ids = [date_id for date_id in unique_date_ids if 850 < date_id <= 1650]
random.shuffle(unique_date_ids)
iterations = len(unique_date_ids)
selected_date_ids = []
# print(unique_date_ids)


test_date_ids = test_date_ids_specific
test = alltraindata.filter(pl.col("date_id").is_in(test_date_ids)).collect()
X_test, y_test, weights_test = get_features(test)
del test
test_X = torch.tensor(X_test,dtype=torch.float32).to(device)
test_y = torch.tensor(y_test, dtype=torch.float32).to(device)
test_weights = torch.tensor(weights_test,dtype=torch.float32).to(device)
test_dataset = CustomDataset(test_X, test_y, test_weights)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
del X_test,y_test,weights_test

for i in range(0,iterations,200):
    selected_date_ids = unique_date_ids[i:i+200]
    # # Separate dates 1690 to 1698 for the test set
    # print iteration 
    print(f"Iteration {i} begining")
    # 

    # # Split remaining date_ids into train and val subsets
    train_size = int(len(selected_date_ids))  # 70% for training

    train_date_ids = selected_date_ids
    

    # Filter data based on date_id splits
    train = alltraindata.filter(pl.col("date_id").is_in(train_date_ids)).collect()
    X_train, y_train, weights_train  = get_features(train)
    del train
    gc.collect()
 
    gc.collect()



    train_X = torch.tensor(X_train,dtype=torch.float16).to(device)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_weights = torch.tensor(weights_train,dtype=torch.float32).to(device)
    train_dataset = CustomDataset(train_X, train_y,train_weights)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    best_model = model
    del X_train, y_train, weights_train
    gc.collect()
    print(f"Training on {len(train_date_ids)} date_ids")
    if evaluate:
        targets,probs,val_mse = evaluate_ae(model,test_loader,device)
    else:
        for epoch in range(epochs):
            train_loss, train_mse = train_ae(model,train_loader, optimizer, loss_function, device)
            
            val_mse  = evaluate_ae(model,test_loader,device)
        
            print(f'epoch {epoch} train loss {train_loss:.4f}, train_mse {train_mse:.4f}, val_mse {val_mse:.4f}')
            del train_loss, train_mse
            if val_mse < best:
                best = val_mse
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), f'{model_save_dir}/{model_name}.pth')
                print(f"Model saved at {model_save_dir}/{model_name}.pth iteration {i} with val_mse {val_mse}")
                degraded = 0
                # plot(targets,probs,[245,455])
            else:
                degraded += 1
            if degraded > 5:
                break
            del val_mse

