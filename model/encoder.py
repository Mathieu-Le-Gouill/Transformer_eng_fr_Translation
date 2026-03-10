from torch import nn, Tensor
from model.multi_head_attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, d_model: int=512, d_ff: int=2048, num_heads: int=6, dropout: float=0.1):
        """
        Transformer Encoder Block

        Args:
            d_model: dimensionality of the token embeddings.
            d_ff: dimensionality of the hidden layer in the feed-forward network.
            num_heads: number of attention heads in the multi-head attention layer.
            dropout: dropout probability applied after attention and feed-forward layers.
        """
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
    
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_model))    
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x: Tensor, mask=None):
        """
        Forward pass of the encoder block.

        Args:
            x: input tensor of shape (B, seq_len, d_model).
            mask: optional attention mask applied during self-attention.

        Returns:
            output tensor of shape (B, seq_len, d_model).
        """

        # Self-attention
        n = self.norm1(x)
        attn, loss_attn = self.attn(n, n, n, mask)
        x = x + self.dropout(attn)

        # Feed-forward
        n = self.norm2(x)
        ff = self.ffn(n)
        x = x + self.dropout(ff)
        
        return x, loss_attn