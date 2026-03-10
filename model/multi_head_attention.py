import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int=768, num_heads: int=3, dropout: float=0.1):
        """
        Multi-Head Attention Module.

        Args:
            d_model: dimensionality of the input embeddings.
            num_heads: number of attention heads.
            dropout: dropout probability applied to attention weights.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)  

        self.dropout = dropout   
        self.attn_dropout = nn.Dropout(dropout) 


    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        """
        Compute the multi-head attention forward pass.

        Args:
            query: query tensor of shape (B, seq_len, d_model).
            key: key tensor of shape (B, key_len, d_model).
            value: value tensor of shape (B, key_len, d_model).
            mask: optional attention mask applied to attention scores.

        Returns:
            output tensor of shape (B, seq_len, d_model).
        """

        batch_size, seq_len, d_model = query.size()
        key_len = key.size(1)

        Q = self.w_query(query) # (B, seq_len, d_model)
        K = self.w_key(key) # (B, key_size, d_model)
        V = self.w_value(value) # (B, key_size, d_model)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2) # (B, num_heads, seq_len, d_head)
        K_T = K.view(batch_size, key_len, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (B, num_heads, d_head, key_len)
        V = V.view(batch_size, key_len, self.num_heads, self.d_head).transpose(1, 2) # (B, num_heads, key_len, d_head)

        out, sparsity_loss = self._scaled_dot_product(Q, K_T, V, mask, self.attn_dropout) # (B, num_heads, seq_len, d_head)

        out = out.transpose(1, 2).contiguous() # (B, seq_len, num_heads, d_head)
        out = out.view(batch_size, seq_len, d_model) # (B, seq_len, d_model)

        out = self.linear_out(out) # (B, self.num_heads, seq_len, d_model)

        return out, sparsity_loss
    
    
    def _scaled_dot_product(self, query: Tensor, key_T: Tensor, value: Tensor, mask=None, dropout=None):
        """
        Compute scaled dot-product attention.

        Args:
            query: tensor of shape (B, num_heads, seq_len, d_head).
            key_T: transposed key tensor of shape (B, num_heads, d_head, key_len).
            value: value tensor of shape (B, num_heads, key_len, d_head).
            mask: optional attention mask applied before softmax.
            dropout: optional dropout applied to attention weights.

        Returns:
            attention output tensor of shape (B, num_heads, seq_len, d_head).
        """

        d_head = query.size()[-1]

        scaled = torch.matmul(query, key_T) / math.sqrt(d_head) # (B, num_heads, seq_len, key_len)

        if mask is not None:
            scaled = scaled.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scaled, dim=-1) # (B, num_heads, seq_len, key_len)

        # --- sparsity loss
        entropy = - (attention * torch.log(attention + 1e-12))
        if mask is not None:
            valid_positions = mask.bool()
            sparsity_loss = entropy.masked_select(valid_positions).mean()
        else:
            sparsity_loss = entropy.mean()


        if dropout is not None:
            attention = dropout(attention)

        out = torch.matmul(attention, value) # (B, num_heads, seq_len, d_head)

        return out, sparsity_loss
