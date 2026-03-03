from torch import nn
from model.multi_head_attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_heads=6, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
    
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_model))    
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    # x: [batch_size, seq_len, d_model]
    def forward(self, x, mask=None):

        # Self-attention
        n = self.norm1(x)
        attn = self.attn(n, n, n, mask)
        x = x + self.dropout(attn)

        # Feed-forward
        n = self.norm2(x)
        ff = self.ffn(n)
        x = x + self.dropout(ff)
        
        return x 