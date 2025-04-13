from torch.utils.data import DataLoader, TensorDataset
from model import Non_AR_TST
from train import *

from datasets import DataSheet
 

device = "cpu"
model = Non_AR_TST.from_config("config.json")  


data = DataSheet(
path="/Users/paarthanimbalkar/Desktop/ri6/data/apple_stock.xlsx",
features=["OPEN,","CLOSE","HIGH","LOW","VOLUME"],
target=["OPEN"],
standardize=True,
seq_len=14,
pred_len=14,
pos="abs",
split_type="stratified",
period_1_end="06-03-2020",
period_2_end="01-01-2023",
train_test_val_split=(0.4, 0.3, 0.3)
)

batch_size = 8


train_loader = DataLoader(TensorDataset(*data.train_split()), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(*data.validate_split()), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(*data.test_split()), batch_size=batch_size, shuffle=False)

trained_model = train(model, train_loader, val_loader, test_loader, config, device)
metrics = evaluate_model(trained_model, train_loader, val_loader, test_loader, data, device)
plot_forecast(trained_model, train_loader, val_loader, test_loader, data, seq_len=data.seq_len, pred_len=data.pred_len, device=device)