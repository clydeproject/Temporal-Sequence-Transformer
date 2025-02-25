import torch 
import torch.nn as nn 
from modules import Projection,SinusoidalPositionalEncoding,MultiHeadAttention,Decoder,FFN

import json 


class TST(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,embed_dim:int,
                 ffn_dim:int,num_heads:int,num_layers:int,
                 seq_len:int,pred_len:int,dropout_p = 0,
                 pe=SinusoidalPositionalEncoding,attn=MultiHeadAttention,nonlinearity=torch.relu,
                ):

        super(TST,self).__init__()
        #config
        self.embed_dim = embed_dim 
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.ffn_dim = ffn_dim 
        self.num_heads = num_heads 
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len 
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p 

        self.input_proj = Projection(self.embed_dim,self.in_dim)
        self.pe = pe(self.embed_dim,dropout_p,)
        self.attn = attn
        #stack of decoder blocks 
        self.layers = nn.Sequential(
            *[Decoder(
                self.embed_dim,
                self.num_heads,
                self.ffn_dim,
                self.seq_len,
                self.attn,
                self.nonlinearity,
                self.dropout_p,

            ) for _ in range(self.num_layers)]
        )

        self.out_proj = nn.Linear(self.embed_dim,self.out_dim)

    def forward(self,x,timestamps,ar=False):
        #not tested
        if ar:
            batch_size, seq_len, _ = x.shape
            # Initial input projection
            z = self.input_proj(x)
            z = self.pe(z, timestamps, mode="add")

            predictions = []
            current_input = z  # Start with the encoded input sequence

            for _ in range(self.pred_len):
                # Pass through the decoder layers
                for layer in self.layers:
                    current_input = layer(current_input)

                # Get the next predicted step (last time step in the sequence)
                next_step = self.out_proj(current_input[:, -1:, :])  # Shape: (batch_size, 1, out_dim)

                predictions.append(next_step)

                # Append prediction to input and shift window
                next_step_embed = self.input_proj(next_step)  # Project prediction into embedding space
                current_input = torch.cat([current_input, next_step_embed], dim=1)[:, -seq_len:, :]

            return torch.cat(predictions, dim=1)  # Shape: (batch_size, pred_len, out_dim)
        #not tested 

        z = self.input_proj(x)
        z = self.pe(z,timestamps,mode="add")
        for layer in self.layers:
            z = layer(z)
        z = z[:, -self.pred_len:, :]#:) 
        z = self.out_proj(z)
        return z
    
    #export model config to json file 
    def export_config(self,path):
        with open(path+"/config.json",mode="w+") as _:
            config = {
                "in_dim" : self.in_dim,
                "out_dim" : self.out_dim,
                "embed_dim" : self.embed_dim,
                "ffn_dim" : self.ffn_dim,
                "num_heads" : self.num_heads,
                "num_layers": self.num_layers,
                "seq_len":self.seq_len,
                "pred_len":self.pred_len,
                "pe":self.pe.__class__.__name__,
                "attn":self.attn.__name__,
                "nonlinearity":self.nonlinearity.__name__,
                "dropout_p":self.dropout_p
            }
            json.dump(config,_,indent=4,)

    #create model directly from config.json file 
    def from_config(config_file_path):
        with open(config_file_path) as _:
            config = json.load(_)
            #improve mapping 
            map_ = {
            "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
            "MultiHeadAttention": MultiHeadAttention,
            "nonlinearity":getattr(torch.nn,config["nonlinearity"])().forward,
            }
            config["pe"] = map_[config["pe"]]
            config["attn"] = map_[config["attn"]]
            config["nonlinearity"] = map_["nonlinearity"]

            return TST(**config)


    def export(self,path):
        pass 


    
