import torch 
import torch.nn as nn 

class AvgPool1d(nn.Module):
    def __init__(self,window_size,):
        super(AvgPool1d,self).__init__()
        self.window_size = window_size
        self.pooler = nn.AvgPool1d(kernel_size=window_size,stride=1)

    def forward(self,t):
        t=t.unsqueeze(0).unsqueeze(0)
        return self.pooler(t).squeeze()


class Projection(nn.Module):
    def __init__(self,embed_dim,in_dim,activation=lambda x: x,dropout_p=0,norm=False):
        super(Projection,self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim 
        self.__norm = norm 
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_p) 
        self.norm = nn.LayerNorm(self.embed_dim,) 

        self.linear_layer = nn.Linear(self.in_dim,self.embed_dim)

    def forward(self,t,):
        x = self.linear_layer(t)
        x = self.activation(x)
        x = self.dropout(x)
        if self.__norm:
            x = self.norm(x)
        return x 
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,embed_dim,dropout_p=0,):
        super(SinusoidalPositionalEncoding,self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z, pos_t, mode="add"):    
        if pos_t.dim() == 0:
            pos_t = pos_t.unsqueeze(0)
        if pos_t.dim() == 2:
            pos_t = pos_t.unsqueeze(-1)

        batch_size, seq_len = z.shape[0], z.shape[1]  
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2, dtype=torch.float32) / self.embed_dim).to(pos_t.device)
        
        pos_t = pos_t.squeeze(-1)  
        pos_enc = torch.zeros(batch_size, seq_len, self.embed_dim, device=pos_t.device) 

        div_term = pos_t.unsqueeze(-1) / denominator
        pos_enc[:, :, 0::2] = torch.sin(div_term)
        pos_enc[:, :, 1::2] = torch.cos(div_term)
        pos_enc.requires_grad_(False)
        if mode == "add":
            z = z + pos_enc
        else:  # concat mode
            z = torch.cat([z, pos_enc.expand(batch_size, -1, -1)], dim=-1)  
        return self.dropout(z)
    

class LearnablePositionEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim, dropout_p=0):
        """
        Args:
        - seq_len: sequence length of the input
        - embed_dim: dimension of the embeddings
        - dropout_p: dropout probability
        """
        super(LearnablePositionEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_p)
        
        #learnable positional embeddings ( matrix of seq+pred len X embed_dim)
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )
        
    def forward(self, z, pos_t=None, mode="add"):
        batch_size, seq_len, embed_dim = z.shape
        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        
        pos_enc = self.pos_embeddings.expand(batch_size, -1, -1)
        
        if mode == "add":
            z = z + pos_enc
        else:  # concat mode
            z = torch.cat([z, pos_enc], dim=-1)
            
        return self.dropout(z)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scaling = float(self.head_dim) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scaling
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.output_proj(out)

class FFN(nn.Module):
    def __init__(self,ffn_dim,embed_dim,dropout_p = 0,activation=torch.relu):
        super(FFN,self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        self.up_proj = nn.Linear(self.embed_dim,self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim,self.embed_dim)
        
    def forward(self,x):
        z = self.up_proj(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.down_proj(z)
        return z
    
class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_dim,attn=MultiHeadAttention,non_linearity=torch.relu,dropout_p=0,):
        super(Decoder,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.non_linearity = non_linearity

        self.multi_attn = attn(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0,
        )
        self.ffn = FFN(
            ffn_dim=self.ffn_dim,
            embed_dim=self.embed_dim,
            dropout_p=0,
            activation=self.non_linearity,
        )

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,x,mask=None):
        attn_out = self.multi_attn(x=x,mask=mask)
        x = self.norm1(x+self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x+self.dropout(ffn_out))
        return x 