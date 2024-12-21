import polars as pl
import numpy as np 
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch


def normalize_dataframe(df: pl.DataFrame, means: dict, stds: dict) -> pl.DataFrame:
    # We normalize the polars dataframe using the provided means and standard deviations
    normalize_exprs = []

    for col in df.columns:
        if col in means and col in stds: #only normalize columns present in the means and std
            if stds[col] != 0: #avoid division by 0
                #Normalize the column and alias it with the same name
                normalize_exprs.append(
                    ((pl.col(col) - means[col]) / stds[col]).alias(col)
                )
            else:
                normalize_exprs.append(pl.col(col) - means[col]).alias(col)

    normalized_df = df.select(normalize_exprs) #Dataframe with normalized expressions
    return normalized_df

def convert_to_vector(date_ids, time_ids):
    date_ids = np.array(date_ids).reshape(-1, 1)
    time_ids = np.array(time_ids).reshape(-1, 1)
    # print(date_ids)/
    # print(data_ids.shape)
    date_vectors = np.hstack((date_ids / 30, date_ids / 12, date_ids / 365))
    # print(date_vectors)
    # print("hrerer")
    time_vectors = np.hstack((time_ids / 16, time_ids / 60, time_ids / 360))
    
    return date_vectors, time_vectors

def get_features(train):
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    train_features = train.select(feature_names)
    train_features = train_features.fill_null(strategy='forward').fill_null(0)
    
    # date_ids,time_ids = convert_to_vector(date_ids, time_ids)
    # train_features = normalize_dataframe(train_features,means,stds)
    X = train_features.to_numpy()
    del train_features
    y = train.select('responder_6').to_numpy().reshape(-1)
    weights = train.select('weight').to_numpy().reshape(-1)

    return X,y,weights

def r2_score(y_true, y_pred, weights):
    """
    Calculate the sample weighted zero-mean R-squared score.

    Parameters:
    y_true (numpy.ndarray): Ground-truth values for responder_6.
    y_pred (numpy.ndarray): Predicted values for responder_6.
    weights (numpy.ndarray): Sample weight vector.

    Returns:
    float: The weighted zero-mean R-squared score.
    """
    numerator = np.sum(weights * (y_true - y_pred)**2)
    denominator = np.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator
    return r2_score


def train_model(model, encoder,loader, optimizer, loss_function, device):
    model.train() #set model to training mode
    total_loss = 0
    all_probs = []
    all_targets = []
    all_weights = []
    # model.noise.training = True
    #Iterate over batches 
    progress_bar = tqdm(loader, desc="Training Progress", leave=True,position =0)
    for X_batch, y_batch, weights_batch in progress_bar:
        #Move data to specified device (CPU or GPU)
        # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # weights_batch = weights_batch.to(device)
        #Reset Gradient to 0
        optimizer.zero_grad()
        # print(batch_time.shape,batch_date.shape)
        # encoded,_ = encoder(X_batch)
        # concat = torch.cat((X_batch, encoded), dim=1)
        # print(X_batch)
        outputs = model(X_batch)
        outputs = outputs.squeeze(dim=1)
        y_batch = y_batch
        loss_per_sample = loss_function(outputs, y_batch)
        weighted_loss = loss_per_sample*weights_batch
        #Compute average loss across the batch
        loss = weighted_loss.mean()
        # print(f"Batch Loss: {loss.item():.4f}")
        progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        loss.backward()

        #Update model parameters
        optimizer.step()

        total_loss += loss.item()

        all_probs.append(outputs.detach().cpu())
        all_targets.append(y_batch.cpu())
        all_weights.append(weights_batch.cpu())
        
    all_probs = torch.cat(all_probs).numpy()   
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()
    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)

    avg_loss = total_loss / len(loader)
    return avg_loss, mse, r2

def evaluate_model(model, encoder,loader,generate_preds = False):
    model.eval()
    all_probs = []
    all_targets = []
    all_weights = []
    # model.noise.training = False
    with torch.no_grad():
        for X_batch, y_batch, weights_batch in tqdm(loader, desc="Validating Progress", leave=True,position=0):
            y_batch = y_batch
            # print(batch_time.shape,batch_date.shape)
            # encoded,_ = encoder(X_batch)
            # concat = torch.cat((X_batch, encoded), dim=1)
            outputs = model(X_batch,False)

            all_probs.append(outputs.squeeze(dim=1).cpu())
            all_targets.append(y_batch.cpu())
            all_weights.append(weights_batch.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()
    
    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)
    print(f"Total Loss val: {mse:.4f} r2 {r2}")
    if generate_preds:
        return all_targets,all_probs,mse,r2
    return mse, r2


