import torch
from torch import nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=3, dropout=0.1):
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

    # query: [batch_size, seq_len, d_model]
    # key:   [batch_size, key_len, d_model]
    # value: [batch_size, key_len, d_model]
    def forward(self, query, key, value, mask=None):

        batch_size, seq_len, d_model = query.size()
        key_len = key.size(1)

        Q = self.w_query(query) # [batch_size, seq_len, d_model]
        K = self.w_key(key) # [batch_size, key_size, d_model]
        V = self.w_value(value) # [batch_size, key_size, d_model]

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2) # [batch_size, num_heads, seq_len, d_head]
        K_T = K.view(batch_size, key_len, self.num_heads, self.d_head).permute(0, 2, 3, 1) # [batch_size, num_heads, d_head, key_len]
        V = V.view(batch_size, key_len, self.num_heads, self.d_head).transpose(1, 2) # [batch_size, num_heads, key_len, d_head]

        out = self._scaled_dot_product(Q, K_T, V, mask, self.attn_dropout) # [batch_size, num_heads, seq_len, d_head]

        out = out.transpose(1, 2).contiguous() # [batch_size, seq_len, num_heads, d_head]
        out = out.view(batch_size, seq_len, d_model) # [batch_size, seq_len, d_model]

        out = self.linear_out(out) # [batch_size, self.num_heads, seq_len, d_model]

        return out
    
    
    # query: [batch_size, num_heads, seq_len, d_head]
    # key:   [batch_size, num_heads, d_head, key_len]
    # value: [batch_size, num_heads, key_len, d_head]
    def _scaled_dot_product(self, query, key_T, value, mask=None, dropout=None):

        d_head = query.size()[-1]

        scaled = torch.matmul(query, key_T) / math.sqrt(d_head) # [batch_size, num_heads, seq_len, key_len]

        if mask is not None:
            scaled = scaled.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scaled, dim=-1) # [batch_size, num_heads, seq_len, key_len]

        if dropout is not None:
            attention = dropout(attention)

        out = torch.matmul(attention, value) # [batch_size, num_heads, seq_len, d_head]

        return out
