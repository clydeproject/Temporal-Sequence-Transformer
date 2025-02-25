import math 
import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset


class CarbonMonitorGlobalIndiaMtCo2(Dataset):
    def __init__(self,path,target_sector,feature_sectors,seq_len,pred_len,standardize=False,train_test_val_split=(.7,.1,.2)):
        self.df = pd.read_excel(path)
        self.countries = set(self.df["country"])
        self.target_sector = target_sector
        self.target_variable = self.df["MtCO2 per day"]
        self.feature_sectors = set(feature_sectors)#all sectors have to be unique 
        self.sectors = set(self.df["sector"])
        self.timesteps = self.df.shape[0]
        
        self.Industry = self.df.groupby("sector").get_group("Industry")
        self.International_Aviation = self.df.groupby("sector").get_group("International Aviation")
        self.Ground_Transport = self.df.groupby("sector").get_group("Ground Transport")
        self.Domestic_Aviation = self.df.groupby("sector").get_group("Domestic Aviation")
        self.Residential = self.df.groupby("sector").get_group("Residential")

        self.x_i = []
        self.y_i = torch.tensor(self.df.groupby("sector").get_group(self.target_sector)["MtCO2 per day"].to_numpy(dtype=np.float32),dtype=torch.float32).T.unsqueeze(1)
        
        for sector in self.sectors:
            if sector in self.feature_sectors:
                self.x_i.append(self.df.groupby("sector").get_group(sector)["MtCO2 per day"].to_numpy(dtype=np.float32))

        self.x_i = torch.tensor(self.x_i,dtype=torch.float32).T
        if standardize:
            self.x_i = ((self.x_i - self.x_i.mean())/self.x_i.std())#~N(0,1)
            self.y_i = ((self.y_i - self.y_i.mean())/self.y_i.std())#~N(0,1)

        self.train_test_val_samples = [int(math.floor(self.x_i.shape[0]*_)) for _ in train_test_val_split]#floors to not exceed samples.(safe bet xd)


    #chronological train/test/val splitting
    def train_split(self,):
        return [self.x_i[:self.train_test_val_samples[0],:],self.y_i[:self.train_test_val_samples[0],:]]

    def test_split(self,):
        return [self.x_i[self.train_test_val_samples[0]:self.train_test_val_samples[0]+self.train_test_val_samples[1], :],
                self.y_i[self.train_test_val_samples[0]:self.train_test_val_samples[0]+self.train_test_val_samples[1], :]]

    def validate_split(self,):
        return [self.x_i[self.train_test_val_samples[0]+self.train_test_val_samples[1]:,:],
                self.y_i[self.train_test_val_samples[0]+self.train_test_val_samples[1]:,:]]


    def granularity(self,):#groked this :) 
        self.df["date"] = pd.to_datetime(self.df["date"], dayfirst=True)
        unique_dates = [pd.Timestamp(date) for date in sorted(self.df["date"].unique())]

        if len(unique_dates) < 2:
            granularity = "Single day data (granularity unclear without more dates)"
            period = "1 day"
            return granularity, period

        date_diffs = [(unique_dates[i+1] - unique_dates[i]) / pd.Timedelta(days=1) for i in range(len(unique_dates)-1)]
        # Determine granularity based on the most common interval
        min_diff = min(date_diffs)  
        if min_diff == 1 and all(diff == 1 for diff in date_diffs):
            granularity = "Daily"
        elif min_diff == 7 and all(diff == 7 for diff in date_diffs):
            granularity = "Weekly"
        elif min_diff >= 28 and min_diff <= 31 and all(diff >= 28 and diff <= 31 for diff in date_diffs):
            granularity = "Monthly (approximate)"
        else:
            granularity = f"Variable intervals (smallest: {min_diff} days)"

        #duration of ts
        start_date = unique_dates[0]
        end_date = unique_dates[-1]
        total_days = (end_date - start_date) / pd.Timedelta(days=1) + 1
        period = f"{int(total_days)} days (from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')})"

        return granularity, period


        def __getitem__(self,):
            pass 

        def __len__(self):
            return self.df.shape 



data = CarbonMonitorGlobalIndiaMtCo2(
    path="/Users/paarthanimbalkar/Desktop/ri6/data/carbon-monitor-carbonmonitorGLOBAL-India.xlsx",
    target_sector="Industry",
    seq_len=100,
    pred_len=100,
    standardize=True,
    feature_sectors=["Domestic Aviation","Ground Transport","International Aviation","Power","Residential"]
    )


print(data.df.head(6))
print(data.countries)
print(data.sectors)
print(data.timesteps)
print(data.granularity())
print(data.Industry)
print(data.feature_sectors)
print(data.x_i.shape)
print(data.y_i.shape)
print(data.y_i)
print(data.Domestic_Aviation)

x_train,y_train = data.train_split()
print(x_train.shape,y_train.shape)
print(data.train_test_val_samples)
