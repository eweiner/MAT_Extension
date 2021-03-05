import torch
from tqdm import tqdm
import copy

from transformer import make_model


def mse_loss(output, y):
    return torch.sum((y - output) ** 2)

def get_mean_error(dl, model, loss_func = mse_loss, tqdm = False):
    model.eval()
    with torch.no_grad():
        n_points = 0
        cum_loss = 0
        for_iter = tqdm(dl) if tqdm else dl
        for batch in for_iter:
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            cum_loss += torch.sum((output - y) ** 2)
            n_points += output.shape[0]
        mse = cum_loss / n_points
    model.train()
    return mse

def train_model(train_loader, test_loader, model, optimizer, epochs, loss_func=mse_loss, replace_every=1, use_tqdm=False, print_epoch_loss=False):
    optimizer.zero_grad()
    training_losses = []
    validation_losses = []
    best_loss = 1e10
    best_model = copy.deepcopy(model)
    validation_loss = get_mean_error(test_loader, best_model)
    print(f"Pre-Training Validation Loss: {validation_loss}")
    for_iter = tqdm(range(epochs)) if use_tqdm else range(epochs)
    for e in for_iter:
        for batch in train_loader:
            optimizer.zero_grad()
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            loss = loss_func(output, y)
            loss /= output.shape[0]
            loss.backward()
            optimizer.step()
            training_losses.append(loss.detach())
        
        if e % replace_every == 0:
            dset_loss = get_mean_error(train_loader, model)
            if dset_loss < best_loss:
                best_loss = dset_loss
                best_model = copy.deepcopy(model)

            validation_loss = get_mean_error(test_loader, best_model)
            if print_epoch_loss:
                print(f"Epoch {e+1} Validation Loss: {validation_loss}")
            validation_losses.append(validation_loss.cpu().numpy())
    return validation_losses

def run_experiment(model_params, train_loader, test_loader, n_trials, epochs=30, use_cuda=False):
    trial_results = []
    for trial in tqdm(range(n_trials)):
        model = make_model(**model_params)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        vl = train_model(train_loader, test_loader, model, optimizer, epochs)
        trial_results.append(vl)
        print(f"Ending experiment {trial} with validation loss {vl[-1]}")
    return trial_results
