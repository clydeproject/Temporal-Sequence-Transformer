import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

#training config 
config = {
    "learning_rate": 1e-4,
    "training_steps": 6000,
    "warmup_steps": 1000,
    "scheduler_type": "cosine",
    "evaluation_steps": 100,
    "bias_norm_weight_decay": False,
    "weight_decay": 0.001,
    "betas": (0.9, 0.98),
    "adam_eps": 1e-6,
}

def lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def mse(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def rmse(predictions, targets):
    return torch.sqrt(mse(predictions, targets))

def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def mape(predictions, targets):
    epsilon = 1e-10
    # Mask out targets close to zero to avoid extreme MAPE values
    mask = torch.abs(targets) > epsilon
    if mask.sum() == 0:  # If all targets are near zero, return NaN or a large value
        return torch.tensor(float('nan'))
    return torch.mean(torch.abs((targets[mask] - predictions[mask]) / (targets[mask] + epsilon))) * 100

def r2_score(predictions, targets):
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_res = torch.sum((targets - predictions) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)

def train(model, train_loader, val_loader, test_loader, config, device):
    criterion = nn.MSELoss()
    decay_params = [param for name, param in model.named_parameters() if "bias" not in name and "norm" not in name]
    no_decay_params = [param for name, param in model.named_parameters() if "bias" in name or "norm" in name]
    optimizer = optim.Adam(
        [{"params": decay_params, "weight_decay": config["weight_decay"]},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=config["learning_rate"],
        betas=config["betas"],
        eps=config["adam_eps"],
    )
    scheduler = lr_scheduler(optimizer, config["warmup_steps"], config["training_steps"])

    print(f"Model device: {next(model.parameters()).device}")
    
    model.train()
    step = 0
    running_loss = 0.0

    for epoch in range(100):
        for batch in train_loader:
            if step >= config["training_steps"]:
                break
            inputs, timestamps, targets = [x.to(device) for x in batch]
            if step == 0:
                print(f"Inputs device: {inputs.device}")
            optimizer.zero_grad()
            outputs = model(inputs, timestamps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            running_loss += loss.item()
            step += 1

            if step % config["evaluation_steps"] == 0:
                avg_train_loss = running_loss / config["evaluation_steps"]
                train_rmse = rmse(outputs, targets)
                
                model.eval()
                val_loss, val_rmse = 0.0, 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_inputs, val_timestamps, val_targets = [x.to(device) for x in val_batch]
                        val_outputs = model(val_inputs, val_timestamps)
                        val_loss += criterion(val_outputs, val_targets).item()
                        val_rmse += rmse(val_outputs, val_targets).item()
                
                val_loss /= len(val_loader)
                val_rmse /= len(val_loader)
                
                test_loss, test_rmse = 0.0, 0.0
                with torch.no_grad():
                    for test_batch in test_loader:
                        test_inputs, test_timestamps, test_targets = [x.to(device) for x in test_batch]
                        test_outputs = model(test_inputs, test_timestamps)
                        test_loss += criterion(test_outputs, test_targets).item()
                        test_rmse += rmse(test_outputs, test_targets).item()
                    test_loss /= len(test_loader)
                    test_rmse /= len(test_loader)
                
                model.train()
                
                print(f"Step {step}/{config['training_steps']}:")
                print(f"  Train Loss: {avg_train_loss:.4f}, RMSE: {train_rmse:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}")
                print(f"  Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

        if step >= config["training_steps"]:
            break
    
    return model



def evaluate_model(model, train_loader, val_loader, test_loader, data, device):
    model.eval()
    metrics = {"train": {}, "val": {}, "test": {}}
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    
    if data.standardize:
        y_mean = data.y_mean.item() if torch.is_tensor(data.y_mean) else data.y_mean
        y_std = data.y_std.item() if torch.is_tensor(data.y_std) else data.y_std
        inverse_transform = lambda x: x * y_std + y_mean
    elif data.normalize:
        y_min = data.y_min.item() if torch.is_tensor(data.y_min) else data.y_min
        y_max = data.y_max.item() if torch.is_tensor(data.y_max) else data.y_max
        inverse_transform = lambda x: x * (y_max - y_min) + y_min
    else:
        inverse_transform = lambda x: x

    with torch.no_grad():
        for split, loader in loaders.items():
            all_preds, all_targets = [], []
            total_loss = 0.0
            criterion = torch.nn.MSELoss()
            
            for batch in loader:
                inputs, timestamps, targets = [x.to(device) for x in batch]
                outputs = model(inputs, timestamps)
                outputs_original = inverse_transform(outputs)
                targets_original = inverse_transform(targets)
                total_loss += criterion(outputs_original, targets_original).item()
                
                all_preds.append(outputs_original.cpu())
                all_targets.append(targets_original.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            metrics[split]["MSE"] = mse(all_preds, all_targets).item()
            metrics[split]["RMSE"] = rmse(all_preds, all_targets).item()
            metrics[split]["MAE"] = mae(all_preds, all_targets).item()
            metrics[split]["MAPE"] = mape(all_preds, all_targets).item()
            metrics[split]["R2"] = r2_score(all_preds, all_targets).item()
            metrics[split]["Loss"] = total_loss / len(loader)
    
    print("\n=== Model Performance Report (Original Scale) ===")
    print(f"{'Split':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE (%)':<12} {'R²':<12} {'Avg Loss':<12}")
    print("-" * 70)
    for split in metrics:
        print(f"{split:<10} {metrics[split]['MSE']:<12.4f} {metrics[split]['RMSE']:<12.4f} "
              f"{metrics[split]['MAE']:<12.4f} {metrics[split]['MAPE']:<12.4f} "
              f"{metrics[split]['R2']:<12.4f} {metrics[split]['Loss']:<12.4f}")
    print("-" * 70)
    
    return metrics



def plot_forecast(model, train_loader, val_loader, test_loader, data, seq_len, pred_len, device):
    model.eval()
   
    if data.normalize:
        target_min = data.y_min.item() if torch.is_tensor(data.y_min) else data.y_min
        target_max = data.y_max.item() if torch.is_tensor(data.y_max) else data.y_max
        scale_fn = lambda x: x * (target_max - target_min) + target_min
    elif data.standardize:
        target_mean = data.y_mean.item() if torch.is_tensor(data.y_mean) else data.y_mean
        target_std = data.y_std.item() if torch.is_tensor(data.y_std) else data.y_std
        scale_fn = lambda x: x * target_std + target_mean
    else:
        scale_fn = lambda x: x  

    full_actual = data.df[data.target].values
    time_axis = np.arange(len(full_actual))
    # Explicitly use float64 to support np.nan
    train_preds = np.full_like(full_actual, np.nan, dtype=np.float64)
    val_preds = np.full_like(full_actual, np.nan, dtype=np.float64)
    test_preds = np.full_like(full_actual, np.nan, dtype=np.float64)

    def process_loader(loader, indices, pred_array):
        for i, (inputs, timestamps, _) in enumerate(loader):
            inputs, timestamps = inputs.to(device), timestamps.to(device)
            outputs = model(inputs, timestamps)
            outputs = scale_fn(outputs.cpu().detach().numpy())  
            batch_size = inputs.size(0)
            for j in range(batch_size):
                if i * loader.batch_size + j >= len(indices):
                    continue
                window_start = indices[i * loader.batch_size + j]
                pred_start = window_start + seq_len
                pred_end = pred_start + pred_len
                if pred_end > len(full_actual):
                    pred_end = len(full_actual)
                pred_length = pred_end - pred_start
                pred_array[pred_start:pred_end] = outputs[j, :pred_length, 0]

    process_loader(train_loader, data.train_indices, train_preds)
    process_loader(val_loader, data.val_indices, val_preds)
    process_loader(test_loader, data.test_indices, test_preds)

    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, full_actual, label='Actual', color='blue')
    plt.plot(time_axis, train_preds, label='Train Pred', color='green')
    plt.plot(time_axis, val_preds, label='Val Pred', color='orange')
    plt.plot(time_axis, test_preds, label='Test Pred', color='red')
    train_end = data.train_indices[-1] + seq_len if len(data.train_indices) > 0 else 0
    val_end = data.val_indices[-1] + seq_len if len(data.val_indices) > 0 else train_end
    plt.axvline(x=train_end, color='gray', linestyle='--', label='Train/Val Split')
    plt.axvline(x=val_end, color='black', linestyle='--', label='Val/Test Split')
    plt.legend()
    plt.grid(True)
    plt.title('Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel(data.target)
    plt.show()

