import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class DataSheet(Dataset):
    def __init__(self, path, target, features, seq_len, pred_len, standardize=False, normalize=False, 
                 train_test_val_split=(0.4, 0.3, 0.3), split_type="stratified", 
                 period_1_end='2017-01-02', period_2_end='2017-12-31', pos="relative"):
        
        self.df = pd.read_csv(path) if path[-3:] == "csv" else pd.read_excel(path)
        self.df = self.df.sort_values("Date").reset_index(drop=True)
        self.features = set(features)
        self.target = target[0] if isinstance(target, list) else target
        self.timesteps = self.df.shape[0]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pos = pos
        self.standardize = standardize
        self.normalize = normalize
        self.split_type = split_type.lower()
        self.train_test_val_split = train_test_val_split

        self.df['Date'] = pd.to_datetime(self.df['Date'], utc=True)

        x_i_raw_list = [self.df[feature].to_numpy(dtype=np.float32) for feature in self.features if feature in self.df.columns]
        self.x_i_raw = torch.tensor(np.stack(x_i_raw_list, axis=1), dtype=torch.float32)
        self.y_i_raw = torch.tensor(self.df[self.target].to_numpy(dtype=np.float32), dtype=torch.float32)

        if self.normalize:
            self.x_min = self.x_i_raw.min(dim=0, keepdim=True)[0]
            self.x_max = self.x_i_raw.max(dim=0, keepdim=True)[0]
            self.y_min = self.y_i_raw.min()
            self.y_max = self.y_i_raw.max()
            self.x_i_raw = (self.x_i_raw - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_i_raw = (self.y_i_raw - self.y_min) / (self.y_max - self.y_min + 1e-8)
            print(f"X min: {self.x_min}")
            print(f"X max: {self.x_max}")
            print(f"Y min: {self.y_min}")
            print(f"Y max: {self.y_max}")
        elif self.standardize:
            self.x_mean = self.x_i_raw.mean(dim=0)
            self.x_std = self.x_i_raw.std(dim=0)
            self.y_mean = self.y_i_raw.mean()
            self.y_std = self.y_i_raw.std()
            self.x_i_raw = (self.x_i_raw - self.x_mean) / (self.x_std + 1e-8)
            self.y_i_raw = (self.y_i_raw - self.y_mean) / (self.y_std + 1e-8)
            print(f"X mean: {self.x_mean}")
            print(f"X std: {self.x_std}")
            print(f"Y mean: {self.y_mean}")
            print(f"Y std: {self.y_std}")

        num_samples = self.timesteps - seq_len - pred_len + 1
        self.x_i = torch.zeros(num_samples, seq_len, self.x_i_raw.shape[1])
        self.y_i = torch.zeros(num_samples, pred_len)
        self.timestamps = torch.zeros(num_samples, seq_len)
        for i in range(num_samples):
            self.x_i[i] = self.x_i_raw[i:i + seq_len]
            self.y_i[i] = self.y_i_raw[i + seq_len:i + seq_len + pred_len]
            if self.pos == "relative":
                self.timestamps[i] = torch.arange(0, seq_len, dtype=torch.float32)
            else:
                self.timestamps[i] = torch.arange(i, i + seq_len, dtype=torch.float32)

        self.y_i = self.y_i.unsqueeze(-1)

        sample_dates = self.df['Date'].iloc[self.seq_len:self.seq_len + num_samples].reset_index(drop=True)
        
        if self.split_type == "stratified":
            period_1_end = pd.to_datetime(period_1_end, utc=True)
            period_2_end = pd.to_datetime(period_2_end, utc=True)

            period_1_mask = sample_dates <= period_1_end
            period_2_mask = (sample_dates > period_1_end) & (sample_dates <= period_2_end)
            period_3_mask = sample_dates > period_2_end

            period_1_indices = np.arange(num_samples)[period_1_mask]
            period_2_indices = np.arange(num_samples)[period_2_mask]
            period_3_indices = np.arange(num_samples)[period_3_mask]

            def split_indices(indices, split_ratios):
                n = len(indices)
                if n == 0:
                    return np.array([]), np.array([]), np.array([])
                train_end = int(n * split_ratios[0])
                val_end = train_end + int(n * split_ratios[1])
                return indices[:train_end], indices[train_end:val_end], indices[val_end:]

            p1_train, p1_val, p1_test = split_indices(period_1_indices, train_test_val_split)
            p2_train, p2_val, p2_test = split_indices(period_2_indices, train_test_val_split)
            p3_train, p3_val, p3_test = split_indices(period_3_indices, train_test_val_split)

            self.train_indices = np.sort(np.concatenate([p1_train, p2_train, p3_train]))
            self.val_indices = np.sort(np.concatenate([p1_val, p2_val, p3_val]))
            self.test_indices = np.sort(np.concatenate([p1_test, p2_test, p3_test]))

        elif self.split_type == "chronological":
            total_samples = num_samples
            train_end = int(total_samples * train_test_val_split[0])
            val_end = train_end + int(total_samples * train_test_val_split[1])
            
            all_indices = np.arange(total_samples)
            self.train_indices = all_indices[:train_end]
            self.val_indices = all_indices[train_end:val_end]
            self.test_indices = all_indices[val_end:]

        else:
            raise ValueError("split_type must be 'stratified' or 'chronological'")

        train_set = set(self.train_indices)
        val_set = set(self.val_indices)
        test_set = set(self.test_indices)
        assert len(train_set & val_set) == 0, "Training and validation sets overlap!"
        assert len(train_set & test_set) == 0, "Training and test sets overlap!"
        assert len(val_set & test_set) == 0, "Validation and test sets overlap!"

        print(f"Train samples: {len(self.train_indices)}, Val samples: {len(self.val_indices)}, Test samples: {len(self.test_indices)}")

    def train_split(self):
        print(f"Train date range: {self.df['Date'].iloc[self.train_indices].min()} to {self.df['Date'].iloc[self.train_indices].max()}")
        return [self.x_i[self.train_indices], self.timestamps[self.train_indices], self.y_i[self.train_indices]]

    def validate_split(self):
        print(f"Validation date range: {self.df['Date'].iloc[self.val_indices].min()} to {self.df['Date'].iloc[self.val_indices].max()}")
        return [self.x_i[self.val_indices], self.timestamps[self.val_indices], self.y_i[self.val_indices]]

    def test_split(self):
        print(f"Test date range: {self.df['Date'].iloc[self.test_indices].min()} to {self.df['Date'].iloc[self.test_indices].max()}")
        return [self.x_i[self.test_indices], self.timestamps[self.test_indices], self.y_i[self.test_indices]]

    def __len__(self):
        return len(self.x_i)

    def __getitem__(self, idx):
        return (self.x_i[idx], self.timestamps[idx], self.y_i[idx])





class ApplStock2018_2024(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class AEPhourly(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class NvidiaStock1999_2024(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class AmazonStock(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class NetflixStock(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class BrkBStock(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class ChinaCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class IndiaCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class UsaCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class IndiaPowerCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class IndiaDomesticAviaCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class IndiaGroundTransCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)

class IndiaPowerCo2(DataSheet):
    def __init__(self, path, target, features, **kwargs):
        super().__init__(path, target, features, **kwargs)
