from torch import nn, Tensor
from model.multi_head_attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, d_model: int=512, d_ff: int=2048, num_heads: int=6, dropout: float=0.1):
        """
        Transformer Decoder Block

        Args:
            d_model: dimensionality of the model embeddings
            d_ff: dimensionality of the feed-forward network hidden layer
            num_heads: number of attention heads in multi-head attention
            dropout: dropout probability applied after each sub-layer
        """
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
    

    def forward(self, x: Tensor, encoder_out: Tensor, self_mask=None, cross_mask=None):
        """
        Forward pass of the decoder block.

        Args:
            x: decoder input tensor of shape (B, seq_len, d_model)
            encoder_out: encoder output tensor of shape (B, key_len, d_model)
            self_mask: optional mask for decoder self-attention
            cross_mask: optional mask for encoder-decoder attention

        Returns:
            output tensor of shape (B, seq_len, d_model)
        """
        
        # Masked self-attention
        n = self.norm1(x)
        attn, loss_self = self.self_attn(n, n, n, self_mask)
        x = x + self.dropout(attn)

        # Cross-attention
        n = self.norm2(x)
        attn, loss_cross = self.cross_attn(n, encoder_out, encoder_out, cross_mask)
        x = x + self.dropout(attn)
        
        # Feed-forward 
        n = self.norm3(x)
        ff = self.ffn(n)
        x = x + self.dropout(ff)
        
        return x, loss_self + loss_cross