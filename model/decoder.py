from torch import nn
from model.multi_head_attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_heads=6, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    # x: [batch_size, seq_len, d_model]
    # encoder_out: [batch_size, key_len, d_model]
    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):
        
        # Masked self-attention
        n = self.norm1(x)
        attn = self.self_attn(n, n, n, self_mask)
        x = x + self.dropout(attn)

        # Cross-attention
        n = self.norm2(x)
        attn = self.cross_attn(n, encoder_out, encoder_out, cross_mask)
        x = x + self.dropout(attn)
        
        # Feed-forward 
        n = self.norm3(x)
        ff = self.ffn(n)
        x = x + self.dropout(ff)
        
        return x