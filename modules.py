import torch 
import torch.nn as nn 


class Projection(nn.Module):
    def __init__(self,embed_dim,in_dim,activation=lambda x: x,dropout_p=0,norm=False):
        super(Projection,self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim 
        self.__norm = norm 
        self.activation = activation#f 
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
    
    

#naive implementation from:- https://arxiv.org/pdf/1706.03762 (Attention is all you need)
#fixed seq_len position embedding(requires grad = False)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,embed_dim,dropout_p=0,):
        super(SinusoidalPositionalEncoding,self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z, pos_t, mode="add"):
        """
        Args:
        - z: (batch_size, seq_len, embedding_dim) -> input tensor
        - pos_t: (batch_size, seq_len) or (batch_size, seq_len, 1) -> positional idx
        - mode: "add" or "concat"
            - "add" -> adds positional encoding to z
            - "concat" -> concatenates positional encoding with z along last dim
        """
        
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



#naive implementation from :- https://arxiv.org/pdf/1706.03762 (Attention is all you need)
#pytorch's inbuilt implementation is prolly faster 
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        #key, query and value matrices 
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scaling = float(self.head_dim) ** -0.5
        #attn = (QK^T)/sqrt(embed_dim) 
        attn = torch.matmul(q, k.transpose(-2, -1)) * scaling
        #auto regressive mask can be applied. 
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))#try 0 if bug
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.output_proj(out)


#standard multilayered perceptron for up proj of attn(t) and down proj of attn(t)
class FFN(nn.Module):
    def __init__(self,ffn_dim,embed_dim,dropout_p = 0,activation=torch.relu):
        super(FFN,self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        #try Projection instead?? 
        self.up_proj = nn.Linear(self.embed_dim,self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim,self.embed_dim)
        
    def forward(self,x):
        z = self.up_proj(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.down_proj(z)
        return z

    
#decoder block 
class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_dim,seqlen,attn=MultiHeadAttention,non_linearity=torch.relu,dropout_p=0):
        super(Decoder,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.seq_len = seqlen
        self.non_linearity = non_linearity
        
        #auto regressive mask(mask out future time-steps)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(1, 1, self.seq_len, self.seq_len)) 
        )

        #attention module 
        self.multi_attn = attn(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0,
        )
        #multilayer perceptron 
        self.ffn = FFN(
            ffn_dim=self.ffn_dim,
            embed_dim=self.embed_dim,
            dropout_p=0,
            activation=self.non_linearity,
        )

        self.norm1 = nn.LayerNorm(self.embed_dim)#layer norm post attn 
        self.norm2 = nn.LayerNorm(self.embed_dim)#layer norm post mlp  
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,x):
        batch_size = x.shape[0]
        attn_mask = self.attn_mask.expand(batch_size, -1, -1, -1)#(batch_size,num_heads,seq_len,seq_len)
        attn_out = self.multi_attn(x=x,mask=attn_mask)
        x = self.norm1(x+self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x+self.dropout(ffn_out))
        return x 
    

class Gate(nn.Module):
    pass 


class Expert(nn.Module):
    pass 




